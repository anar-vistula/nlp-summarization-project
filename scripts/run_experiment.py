#!/usr/bin/env python
"""
Run an experiment: every model over every document, with full accounting.

Unlike the interactive Experiment tab, this runner is deliberately NOT
language-gated. Each model is given every document, including the mismatched
pairs (English-only BART on Turkish text, Turkish mBART on English), and each
result is labelled in_language / multilingual / cross_language so the two
regimes can be analysed separately. Those mismatched cells are evidence, not
mistakes.

Everything a run produces is written to one self-contained folder, so results
can be read, cited and re-analysed long after the process exits:

    experiments/<name>/
        manifest.json          configuration, machine, model set, versions
        corpus.jsonl           snapshot of the exact documents used
        runs.jsonl             one line per document x model (the resume index)
        docs/<doc_id>/
            _source.txt        the document as the models received it
            _reference.txt     the human reference summary
            <model>.txt        that model's summary, with a readable header
            scores.json        every metric, for every model, on this document
        logs/run.log

The run is resumable: finished document x model pairs are skipped, so an
interrupted overnight run continues where it stopped.

Examples
--------
    # What would this cost and how long would it take?
    python scripts/run_experiment.py --name pilot --corpus corpus/en.jsonl --limit 5 --dry-run

    # Local models only, no money spent
    python scripts/run_experiment.py --name pilot --corpus corpus/en.jsonl corpus/tr.jsonl --limit 5

    # Full run including paid APIs, with a hard spending cap
    python scripts/run_experiment.py --name main_100 --corpus corpus/en.jsonl corpus/tr.jsonl \
        --paid --max-cost 2.00
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from backend.comparison import ComparisonService  # noqa: E402
from backend.evaluation.text_utils import count_words  # noqa: E402
from backend.evaluation.costs import (  # noqa: E402
    UsageTimer,
    build_usage,
    estimate_api_cost,
    machine_fingerprint,
)
from backend.evaluation.experiment import AGGREGATED_METRICS  # noqa: E402
from backend.evaluation.metrics import SummaryEvaluator  # noqa: E402
from backend.evaluation.model_registry import BY_KEY, language_match, resolve  # noqa: E402

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #

def load_corpus(paths: List[str], limit: Optional[int], per_language: bool = True) -> List[Dict]:
    """Load documents from JSONL files, optionally capped per language."""
    documents: List[Dict] = []
    for path in paths:
        file = Path(path)
        if not file.exists():
            raise SystemExit(f"corpus file not found: {path} (build it with scripts/build_corpus.py)")
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                documents.append(json.loads(line))
    if limit:
        if per_language:
            capped: List[Dict] = []
            counts: Dict[str, int] = {}
            for document in documents:
                lang = document["lang"]
                if counts.get(lang, 0) < limit:
                    capped.append(document)
                    counts[lang] = counts.get(lang, 0) + 1
            documents = capped
        else:
            documents = documents[:limit]
    return documents


# --------------------------------------------------------------------------- #
# Cost forecasting
# --------------------------------------------------------------------------- #

def count_tokens(text: str) -> int:
    """Token count for cost forecasting; falls back to a per-language ratio."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return int(len(text.split()) * 1.6)


def forecast(documents: List[Dict], models: List[Dict], ratio: float) -> Dict:
    """Estimated API spend and (very rough) wall time, before anything runs."""
    # Measured on this machine: seconds per 1000 words, per model type.
    seconds_per_1k_words = {"extractive": 0.05, "hf": 7.0, "colab": 12.0, "paid": 8.0}
    total_cost = 0.0
    per_model = []
    for model in models:
        cost = 0.0
        seconds = 0.0
        for document in documents:
            words = document["words"]
            seconds += seconds_per_1k_words[model["type"]] * (words / 1000)
            if model["type"] == "paid":
                model_id = _paid_model_id(model["key"])
                input_tokens = count_tokens(document["text"])
                output_tokens = min(1000, int(words * ratio * 1.6))
                cost += estimate_api_cost(model_id, input_tokens, output_tokens)
        total_cost += cost
        per_model.append({
            "key": model["key"], "type": model["type"],
            "estimated_cost_usd": round(cost, 4),
            "estimated_minutes": round(seconds / 60, 1),
        })
    return {"total_cost_usd": round(total_cost, 4), "per_model": per_model}


def _paid_model_id(key: str) -> str:
    from backend.summarizers.paid_models import PaidModelSummarizer

    return PaidModelSummarizer.SUPPORTED_MODELS[key]["model_id"]


# --------------------------------------------------------------------------- #
# Experiment folder
# --------------------------------------------------------------------------- #

class ExperimentsFolderPeek:
    """Read-only view of an existing run, used by --dry-run to report progress."""

    def __init__(self, name: str):
        self.total = 0
        self.spent = 0.0
        self.by_model: Dict[str, int] = {}
        path = EXPERIMENTS_DIR / name / "runs.jsonl"
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            self.spent += record.get("api_cost_usd") or 0.0
            if record.get("ok"):
                self.total += 1
                key = record["model"]
                self.by_model[key] = self.by_model.get(key, 0) + 1


class ExperimentFolder:
    """Everything one experiment writes to disk."""

    def __init__(self, name: str):
        self.root = EXPERIMENTS_DIR / name
        self.docs = self.root / "docs"
        self.logs = self.root / "logs"
        for directory in (self.root, self.docs, self.logs):
            directory.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.root / "runs.jsonl"
        self.log_path = self.logs / "run.log"

    def completed_pairs(self) -> Set[Tuple[str, str]]:
        """(document, model) pairs already recorded, so a rerun can resume."""
        done: Set[Tuple[str, str]] = set()
        if self.runs_path.exists():
            for line in self.runs_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("ok"):
                        done.add((record["doc_id"], record["model"]))
        return done

    def spent_so_far(self) -> float:
        total = 0.0
        if self.runs_path.exists():
            for line in self.runs_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    total += json.loads(line).get("api_cost_usd") or 0.0
        return total

    def write_manifest(self, manifest: Dict) -> None:
        """
        Merged rather than overwritten, for the same reason as the corpus snapshot.

        A resume invokes the runner with only what is still missing — typically
        one model and one language — so writing that configuration verbatim
        leaves the experiment's provenance record describing the last resume
        instead of the experiment. That defect reached the report of
        2026-10-27, whose header claimed "100 documents {'en': 100}" and a
        single model for a 200-document, ten-model study. The data was intact;
        the record of what produced it was not.

        Document counts are recomputed from the merged corpus snapshot on disk
        rather than from this invocation's arguments, so they describe the
        experiment and cannot drift from the documents the results refer to.
        Each invocation is also appended to `sessions`, which is what makes an
        interrupted multi-day run auditable.
        """
        path = self.root / "manifest.json"
        merged = dict(manifest)
        if path.exists():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
            if previous:
                merged = {**previous, **manifest}
                merged["created_at"] = previous.get("created_at", manifest.get("created_at"))
                merged["corpus_files"] = sorted(
                    set(previous.get("corpus_files") or []) | set(manifest.get("corpus_files") or [])
                )
                by_key = {m["key"]: m for m in (previous.get("models") or [])}
                by_key.update({m["key"]: m for m in (manifest.get("models") or [])})
                merged["models"] = sorted(by_key.values(), key=lambda m: m["key"])
                merged["sessions"] = list(previous.get("sessions") or [])

        merged.setdefault("sessions", [])
        merged["sessions"].append({
            "at": manifest.get("created_at"),
            "corpus_files": manifest.get("corpus_files"),
            "models": [m["key"] for m in (manifest.get("models") or [])],
            "estimate": manifest.get("estimate"),
        })
        merged["last_extended_at"] = manifest.get("created_at")

        snapshot = self.root / "corpus.jsonl"
        if snapshot.exists():
            by_lang: Dict[str, int] = {}
            total = 0
            for line in snapshot.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    total += 1
                    lang = json.loads(line).get("lang", "?")
                    by_lang[lang] = by_lang.get(lang, 0) + 1
            merged["documents"] = total
            merged["documents_by_language"] = dict(sorted(by_lang.items()))

        path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    def snapshot_corpus(self, documents: List[Dict]) -> None:
        """
        The exact documents used, so results stay reproducible.

        Merged rather than overwritten: an experiment is often extended with a
        second corpus (another language) in a later session, and replacing the
        snapshot would erase the documents the earlier results refer to.
        """
        path = self.root / "corpus.jsonl"
        merged: Dict[str, Dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    existing = json.loads(line)
                    merged[existing["id"]] = existing
        for document in documents:
            merged[document["id"]] = document
        with path.open("w", encoding="utf-8") as handle:
            for document in merged.values():
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")

    def write_document(self, document: Dict) -> None:
        folder = self.docs / document["id"]
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / "_source.txt"
        if not source.exists():
            header = (
                f"# {document['title']}\n"
                f"# id={document['id']} lang={document['lang']} source={document['source']} "
                f"words={document['words']}\n"
                f"# {document.get('source_ref', '')}\n\n"
            )
            source.write_text(header + document["text"], encoding="utf-8")
            (folder / "_reference.txt").write_text(
                f"# reference summary ({document['reference_words']} words)\n\n" + document["reference"],
                encoding="utf-8",
            )

    def write_summary(self, document: Dict, model: Dict, summary: str, flat: Dict, usage) -> None:
        """One readable file per model per document, for qualitative reading."""
        folder = self.docs / document["id"]
        folder.mkdir(parents=True, exist_ok=True)
        header = (
            f"# {model['name']}  ({model['key']})\n"
            f"# document: {document['id']} [{document['lang']}] {document['title'][:60]}\n"
            f"# language match: {flat.get('language_match')}   "
            f"trained for: {model['trained_for']}\n"
            f"# quality={_pct(flat.get('overall_quality'))} rougeL={_pct(flat.get('rouge_l_f1'))} "
            f"faithful={_pct(flat.get('factual_consistency'))} novel={_pct(flat.get('abstractiveness'))} "
            f"redundancy={_pct(flat.get('redundancy'))}\n"
            f"# {len(summary.split())} words, {usage.wall_seconds:.1f}s, "
            f"${usage.total_cost_usd:.6f}\n\n"
        )
        (folder / f"{model['key']}.txt").write_text(header + summary, encoding="utf-8")

    def update_scores(self, document: Dict, model_key: str, metrics: Dict) -> None:
        """Full metric dictionaries for one document, keyed by model."""
        path = self.docs / document["id"] / "scores.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        trimmed = {k: v for k, v in metrics.items() if k not in ("metric_metadata",)}
        payload[model_key] = trimmed
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_run(self, record: Dict) -> None:
        with self.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log(self, message: str) -> None:
        stamped = f"{datetime.now().strftime('%H:%M:%S')} {message}"
        print(stamped, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

def ratio_for(document: Dict, args: argparse.Namespace) -> float:
    """
    Compression ratio for one document.

    With --target-words every model is asked for the same summary length, so
    metrics that scale with length (coverage, information density) compare
    like with like. Otherwise the fixed --ratio applies.
    """
    if not args.target_words:
        return args.ratio
    return max(0.02, min(0.9, args.target_words / max(document["words"], 1)))


def run(args: argparse.Namespace) -> int:
    documents = load_corpus(args.corpus, args.limit)
    if not documents:
        raise SystemExit("no documents loaded")

    resolution = resolve(
        requested=args.models,
        include_paid=args.paid,
        include_colab=not args.no_colab,
        check_colab=not args.no_colab,
    )
    models = resolution["models"]
    if not models:
        raise SystemExit("no runnable models: " + json.dumps(resolution["skipped"], ensure_ascii=False))

    if args.prefer_colab and not args.no_colab:
        from backend.summarizers.paid_models import ColabGPUClient

        hosted = set(ColabGPUClient().get_available_models())
        moved = []
        for model in models:
            if model["type"] == "hf" and model["key"] in hosted:
                model["type"] = "colab"
                moved.append(model["key"])
        if moved:
            print(f"  routed to GPU: {', '.join(moved)}")
        elif hosted:
            print(f"  --prefer-colab: server hosts {sorted(hosted)}, none of them local models")
        else:
            print("  --prefer-colab: Colab server not reachable, running locally")

    estimate = forecast(documents, models, args.ratio if not args.target_words else 0.15)
    languages: Dict[str, int] = {}
    for document in documents:
        languages[document["lang"]] = languages.get(document["lang"], 0) + 1

    print(f"\nExperiment: {args.name}")
    print(f"  documents : {len(documents)}  {languages}")
    print(f"  models    : {len(models)} -> {[m['key'] for m in models]}")
    if resolution["skipped"]:
        for skipped in resolution["skipped"]:
            print(f"    skipped {skipped['key']}: {skipped['reason']}")
    print(f"  runs      : {len(documents) * len(models)}")
    print(f"  estimated : ${estimate['total_cost_usd']:.4f} API, "
          f"{sum(m['estimated_minutes'] for m in estimate['per_model']):.0f} min wall (rough)")
    for entry in estimate["per_model"]:
        if entry["estimated_cost_usd"] or entry["estimated_minutes"] > 1:
            print(f"    {entry['key']:22} ${entry['estimated_cost_usd']:.4f}  ~{entry['estimated_minutes']:.0f} min")

    if args.dry_run:
        existing = ExperimentsFolderPeek(args.name)
        if existing.total:
            print(f"  already done : {existing.total} runs "
                  f"({len(documents) * len(models) - existing.total} remaining), "
                  f"${existing.spent:.4f} already spent")
            for key, count in sorted(existing.by_model.items()):
                print(f"    {key:22} {count}")
        print("\ndry run: nothing executed")
        return 0
    if estimate["total_cost_usd"] > args.max_cost:
        raise SystemExit(
            f"\nestimated API cost ${estimate['total_cost_usd']:.2f} exceeds --max-cost "
            f"${args.max_cost:.2f}. Raise the cap deliberately if that is intended."
        )

    folder = ExperimentFolder(args.name)
    done = folder.completed_pairs()
    already_spent = folder.spent_so_far()
    if done:
        print(f"  resuming  : {len(done)} runs already complete, ${already_spent:.4f} already spent")

    folder.snapshot_corpus(documents)
    folder.write_manifest({
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_files": args.corpus,
        "documents": len(documents),
        "documents_by_language": languages,
        "models": [{k: m[k] for k in ("key", "name", "type", "trained_for")} for m in models],
        "skipped_models": resolution["skipped"],
        "ratio": args.ratio,
        "target_words": args.target_words,
        "length_control": "fixed target words" if args.target_words else "fixed ratio",
        "cross_lingual": True,
        "estimate": estimate,
        "machine": machine_fingerprint(),
    })
    for document in documents:
        folder.write_document(document)

    service = ComparisonService()
    evaluator = SummaryEvaluator()
    # Circuit breaker: when a Colab session expires mid-run, every remaining
    # request fails instantly. Without this the runner would burn through the
    # rest of the corpus recording errors in seconds. Stopping instead keeps
    # the log clean and makes the cause obvious.
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5
    aborted = False
    spent = already_spent
    completed = failed = skipped = 0
    total_runs = len(documents) * len(models)
    started = time.perf_counter()

    try:
        # Model-major order: each local model is loaded once, and its pipelines
        # are released before the next one, keeping peak memory to one model.
        for model in models:
            if aborted:
                break
            folder.log(f"=== {model['name']} ({model['key']}) over {len(documents)} documents")
            for index, document in enumerate(documents, 1):
                if (document["id"], model["key"]) in done:
                    skipped += 1
                    continue
                if model["type"] == "paid" and spent >= args.max_cost:
                    folder.log(f"  spending cap ${args.max_cost} reached; skipping remaining paid runs")
                    break

                match = language_match(model["key"], document["lang"])
                try:
                    doc_ratio = ratio_for(document, args)
                    with UsageTimer() as timer:
                        produced = service.run_model(model, document["text"], doc_ratio, document["lang"])
                    usage = build_usage(
                        model["type"], timer,
                        usage_payload=produced.get("usage"),
                        model_id=produced.get("model_id") if model["type"] == "paid" else None,
                        gpu_seconds=produced.get("gpu_seconds"),
                    )
                    metrics = evaluator.evaluate(
                        document["text"], produced["summary"],
                        reference_summary=document.get("reference"), lang=document["lang"],
                    )
                    metrics["_runtime_seconds"] = usage.wall_seconds

                    flat = {
                        name: spec["accessor"](metrics)
                        for name, spec in AGGREGATED_METRICS.items()
                    }
                    flat["language_match"] = match
                    record = {
                        "doc_id": document["id"], "doc_lang": document["lang"],
                        "doc_source": document["source"], "doc_words": document["words"],
                        "model": model["key"], "model_name": model["name"],
                        "model_type": model["type"], "trained_for": model["trained_for"],
                        "language_match": match, "ok": True, "error": None,
                        "ratio": round(doc_ratio, 4),
                        "chunks": produced.get("chunks"),
                        "summary_words": count_words(produced["summary"]),
                        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        **usage.to_dict(),
                        **{k: v for k, v in flat.items() if k != "language_match"},
                    }
                    folder.append_run(record)
                    folder.write_summary(document, model, produced["summary"], flat, usage)
                    folder.update_scores(document, model["key"], metrics)
                    spent += usage.api_cost_usd
                    completed += 1
                    consecutive_failures = 0
                    folder.log(
                        f"  [{completed + skipped + failed}/{total_runs}] {document['id']:22} "
                        f"{match:14} q={_pct(flat.get('overall_quality'))} "
                        f"faith={_pct(flat.get('factual_consistency'))} "
                        f"{usage.wall_seconds:6.1f}s ${usage.total_cost_usd:.5f}"
                    )
                except Exception as error:
                    failed += 1
                    folder.append_run({
                        "doc_id": document["id"], "doc_lang": document["lang"],
                        "model": model["key"], "model_name": model["name"],
                        "model_type": model["type"], "trained_for": model["trained_for"],
                        "language_match": match, "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    })
                    folder.log(f"  FAILED {document['id']} / {model['key']}: {type(error).__name__}: {str(error)[:120]}")
                    if args.debug:
                        traceback.print_exc()
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        folder.log(
                            f"\n  ABORTING: {consecutive_failures} consecutive failures. "
                            f"The GPU server is most likely gone (expired Colab session). "
                            f"Restart Colab, then rerun the same command to resume."
                        )
                        aborted = True
                        break

            # Release this model's weights before loading the next one.
            if model["type"] == "hf":
                service.abstractive._pipelines.clear()
                try:
                    import gc, torch
                    gc.collect()
                    if hasattr(torch, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass
    except KeyboardInterrupt:
        folder.log("\ninterrupted: progress is saved, rerun the same command to resume")

    elapsed = (time.perf_counter() - started) / 60
    folder.log(
        f"\ndone: {completed} completed, {failed} failed, {skipped} already present, "
        f"{elapsed:.1f} min, ${spent:.4f} API spend total"
    )
    print(f"\nresults in {folder.root}")
    print(f"aggregate with: ./venv/bin/python scripts/aggregate.py {folder.root}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="experiment folder name under experiments/")
    parser.add_argument("--corpus", nargs="+", required=True, help="one or more corpus JSONL files")
    parser.add_argument("--models", nargs="+", help="model keys (default: everything runnable)")
    parser.add_argument("--limit", type=int, help="cap documents PER LANGUAGE")
    parser.add_argument("--ratio", type=float, default=0.3, help="target compression ratio")
    parser.add_argument("--target-words", type=int, default=None,
                        help="aim every model at the same summary length (words) by deriving a "
                             "per-document ratio. Without it, a fixed ratio makes long documents "
                             "produce long extractive summaries and short neural ones, and "
                             "length-sensitive metrics reward whichever is longer.")
    parser.add_argument("--paid", action="store_true", help="enable paid API models (costs money)")
    parser.add_argument("--no-colab", action="store_true", help="skip Colab GPU models")
    parser.add_argument("--prefer-colab", action="store_true",
                        help="run every model the Colab server hosts on the GPU instead of "
                             "locally. Keeps the laptop idle: it then only runs TextRank and "
                             "the metric computation.")
    parser.add_argument("--max-cost", type=float, default=1.0, help="hard cap on API spend, USD")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and cost estimate, run nothing")
    parser.add_argument("--debug", action="store_true", help="print tracebacks for failures")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
