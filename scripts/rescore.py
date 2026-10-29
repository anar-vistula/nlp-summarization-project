#!/usr/bin/env python
"""
Recompute metrics for an existing experiment from the stored summaries.

No model is invoked: every summary is already on disk, so a metric fix can be
applied to a finished run without spending GPU time or API credit again. This
is the practical payoff of storing each summary as a file rather than only its
score.

Timing and cost fields are preserved exactly — only metric values change — so
the efficiency analysis stays valid.

    python scripts/rescore.py experiments/main_run
    python scripts/rescore.py experiments/main_run --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.experiment import AGGREGATED_METRICS  # noqa: E402
from backend.evaluation.metrics import SummaryEvaluator  # noqa: E402


def read_summary(path: Path) -> str:
    """Strip the '#' header written by the runner and return the summary text."""
    lines = path.read_text(encoding="utf-8").splitlines()
    body = [l for l in lines if not l.startswith("#")]
    return "\n".join(body).strip()


def rescore(folder: Path, dry_run: bool = False) -> None:
    corpus_path = folder / "corpus.jsonl"
    runs_path = folder / "runs.jsonl"
    if not corpus_path.exists() or not runs_path.exists():
        raise SystemExit(f"{folder} does not look like an experiment folder")

    documents: Dict[str, Dict] = {}
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc = json.loads(line)
            documents[doc["id"]] = doc

    runs = [json.loads(l) for l in runs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    evaluator = SummaryEvaluator()

    changed = 0
    missing = 0
    deltas = []
    new_runs = []
    per_doc_scores: Dict[str, Dict] = {}

    for record in runs:
        if not record.get("ok"):
            new_runs.append(record)
            continue
        doc_id, model = record["doc_id"], record["model"]
        document = documents.get(doc_id)
        summary_file = folder / "docs" / doc_id / f"{model}.txt"
        if not document or not summary_file.exists():
            missing += 1
            new_runs.append(record)
            continue

        summary = read_summary(summary_file)
        metrics = evaluator.evaluate(
            document["text"], summary,
            reference_summary=document.get("reference"), lang=document["lang"],
        )
        metrics["_runtime_seconds"] = record.get("wall_seconds")

        before = record.get("factual_consistency")
        updated = dict(record)
        for name, spec in AGGREGATED_METRICS.items():
            value = spec["accessor"](metrics)
            if value is not None:
                updated[name] = value
        after = updated.get("factual_consistency")
        if before is not None and after is not None and abs(after - before) > 1e-9:
            changed += 1
            deltas.append((model, after - before))

        new_runs.append(updated)
        per_doc_scores.setdefault(doc_id, {})[model] = {
            k: v for k, v in metrics.items() if k != "metric_metadata"
        }

    print(f"rescored {len(new_runs)} rows | faithfulness changed on {changed} | "
          f"missing summaries: {missing}")
    if deltas:
        by_model: Dict[str, list] = {}
        for model, delta in deltas:
            by_model.setdefault(model, []).append(delta)
        print("\nmean change in faithfulness:")
        for model, values in sorted(by_model.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
            print(f"  {model:18} {sum(values) / len(values) * 100:+6.1f} pts  ({len(values)} runs)")

    if dry_run:
        print("\ndry run: nothing written")
        return

    backup = runs_path.with_suffix(".jsonl.bak")
    shutil.copy2(runs_path, backup)
    with runs_path.open("w", encoding="utf-8") as handle:
        for record in new_runs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    for doc_id, scores in per_doc_scores.items():
        (folder / "docs" / doc_id / "scores.json").write_text(
            json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"\nwritten. previous runs.jsonl saved as {backup.name}")
    print("re-run scripts/aggregate.py to refresh the report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rescore(Path(args.folder), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
