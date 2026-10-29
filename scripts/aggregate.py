#!/usr/bin/env python
"""
Aggregate an experiment folder into tables, statistics and a readable report.

Reads experiments/<name>/runs.jsonl and writes into experiments/<name>/aggregate/:

    per_run.csv     every document x model row (the raw data for the appendix)
    summary.csv     per model per document-language: mean, 95% CI, n
    cross_lingual.csv   each model's in-language vs cross-language performance
    cost.csv        API spend, compute time and notional cost per model
    report.md       the human-readable write-up of all of the above

The report is what to read first; the CSVs are for re-analysis and for the
thesis appendix. Nothing here re-runs a model: aggregation is cheap and can be
repeated as often as you like, including while a run is still going.

    python scripts/aggregate.py experiments/pilot
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.statistics import (  # noqa: E402
    MIN_N_FOR_SIGNIFICANCE,
    bootstrap_mean_ci,
    compare_systems,
)

# Metrics reported in the summary table, in order, with their direction.
METRICS = [
    ("overall_quality", "Composite", "higher"),
    ("rouge_l_f1", "ROUGE-L", "higher"),
    ("rouge_1_f1", "ROUGE-1", "higher"),
    ("bert_score_f1", "BERTScore", "higher"),
    ("meteor", "METEOR", "higher"),
    ("information_density", "Density", "higher"),
    ("topic_coverage", "Topics", "higher"),
    ("source_coverage", "Coverage", "higher"),
    ("factual_consistency", "Faithful", "higher"),
    ("redundancy", "Redundancy", "lower"),
    ("abstractiveness", "Novel words", "descriptive"),
    ("extractive_density", "Extr. density", "descriptive"),
    ("compression_ratio", "Compression", "lower"),
    ("runtime_seconds", "Runtime (s)", "lower"),
]
HEADLINE = ["overall_quality", "rouge_l_f1", "bert_score_f1", "factual_consistency",
            "information_density"]
# Metrics that are plain numbers, not proportions: rendering them as percentages
# turns an extractive density of 44.8 into "4476.6%".
PLAIN_NUMBER_METRICS = {"runtime_seconds", "extractive_density"}


def load_runs(folder: Path) -> List[Dict]:
    path = folder / "runs.jsonl"
    if not path.exists():
        raise SystemExit(f"no runs.jsonl in {folder}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: Optional[float], digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def num(value: Optional[float], digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_csv(path: Path, rows: List[Dict], columns: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_degenerate(run: Dict) -> bool:
    """
    A run that completed without error but produced no summary.

    These are not failures — the model returned, so the runner recorded
    `ok: True` — and they are not measurements either. An empty string
    satisfies every containment- and similarity-based metric vacuously:
    faithfulness 100% (nothing to contradict), lexical cohesion 100% (nothing
    incoherent), redundancy 0% (nothing repeated), composing to 25.0%, which
    is ABOVE the mean of the system that produced them on documents where it
    produced text. Left in, they raise the score of the system that failed.
    """
    return (run.get("summary_words") or 0) == 0


def aggregate(folder: Path) -> None:
    runs = load_runs(folder)
    completed = [r for r in runs if r.get("ok")]
    failures = [r for r in runs if not r.get("ok")]
    degenerate = [r for r in completed if is_degenerate(r)]
    ok_runs = [r for r in completed if not is_degenerate(r)]
    if not ok_runs:
        raise SystemExit("no successful runs to aggregate")

    out = folder / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8")) \
        if (folder / "manifest.json").exists() else {}
    model_names = {r["model"]: r.get("model_name", r["model"]) for r in runs}

    # ---------------------------------------------------------------- per_run
    columns = ["doc_id", "doc_lang", "doc_source", "doc_words", "model", "model_name",
               "model_type", "trained_for", "language_match", "summary_words",
               "wall_seconds", "cpu_seconds", "gpu_seconds", "chunks", "input_tokens", "output_tokens",
               "api_cost_usd", "compute_cost_usd", "total_cost_usd"] + [m[0] for m in METRICS]
    write_csv(out / "per_run.csv", ok_runs, columns)

    # ---------------------------------------------------- per language summary
    languages = sorted({r["doc_lang"] for r in ok_runs})
    summary_rows: List[Dict] = []
    stats_by_language: Dict[str, Dict] = {}

    for lang in languages:
        subset = [r for r in ok_runs if r["doc_lang"] == lang]
        # Documents on which some system produced nothing are removed from the
        # comparison entirely, rather than being scored as if an empty summary
        # were a summary. Paired tests require every system on one identical
        # document set, so a document is either in for all systems or out for
        # all. The rejected alternative was to score degenerate runs as zero:
        # that keeps n but invents a value for a measurement never made, and
        # the choice of value would itself drive the result.
        degenerate_docs = {r["doc_id"] for r in degenerate if r["doc_lang"] == lang}
        docs_in_lang = sorted({r["doc_id"] for r in subset} - degenerate_docs)
        subset = [r for r in subset if r["doc_id"] in set(docs_in_lang)]
        models = sorted({r["model"] for r in subset})

        # Only models that completed every document of this language can be
        # compared: otherwise the means rest on different document sets.
        complete = [
            m for m in models
            if {r["doc_id"] for r in subset if r["model"] == m} == set(docs_in_lang)
        ]
        incomplete = [m for m in models if m not in complete]

        per_metric: Dict[str, Dict] = {}
        for metric, _, direction in METRICS:
            scores = {}
            for model in complete:
                values = [
                    r.get(metric) for r in subset
                    if r["model"] == model and r.get(metric) is not None
                ]
                if len(values) == len(docs_in_lang):
                    scores[model] = [float(v) for v in values]
            if not scores:
                continue
            stats = compare_systems(scores) if metric in HEADLINE else {
                "per_system": {m: bootstrap_mean_ci(v) for m, v in scores.items()},
                "pairwise": {},
                "n_documents": len(docs_in_lang),
            }
            stats["direction"] = direction
            per_metric[metric] = stats

            for model, entry in stats["per_system"].items():
                summary_rows.append({
                    "doc_lang": lang, "model": model, "model_name": model_names.get(model, model),
                    "metric": metric, "direction": direction,
                    "mean": entry["mean"], "ci_low": entry["ci_low"], "ci_high": entry["ci_high"],
                    "std": entry["std"], "n": entry["n"],
                })

        stats_by_language[lang] = {
            "documents": docs_in_lang, "models": complete,
            "incomplete": incomplete, "metrics": per_metric,
        }

    write_csv(out / "summary.csv", summary_rows,
              ["doc_lang", "model", "model_name", "metric", "direction",
               "mean", "ci_low", "ci_high", "std", "n"])

    # ------------------------------------------------------------- pairwise
    # Written out so that nothing downstream has to recompute a significance
    # test. The bootstrap is seeded, but resampling a differently ordered list
    # of the same scores draws different resamples, so a second computation
    # yields a slightly different p for the same comparison. One source only.
    pairwise_rows = []
    for lang, entry in stats_by_language.items():
        for metric, stats in entry["metrics"].items():
            for label, r in (stats.get("pairwise") or {}).items():
                pairwise_rows.append({
                    "doc_lang": lang, "metric": metric, "comparison": label,
                    "system_a": r["system_a"], "system_b": r["system_b"],
                    "mean_difference": r["mean_difference"],
                    "diff_ci_low": r.get("diff_ci_low"), "diff_ci_high": r.get("diff_ci_high"),
                    "bootstrap_p": r.get("bootstrap_p"), "p_adjusted": r.get("p_adjusted"),
                    "cohens_d": r.get("cohens_d"), "effect_size": r.get("effect_size"),
                    "significant": r.get("significant"),
                    "n_documents": stats.get("n_documents"),
                })
    if pairwise_rows:
        write_csv(out / "pairwise.csv", pairwise_rows,
                  ["doc_lang", "metric", "comparison", "system_a", "system_b",
                   "mean_difference", "diff_ci_low", "diff_ci_high", "bootstrap_p",
                   "p_adjusted", "cohens_d", "effect_size", "significant", "n_documents"])

    # ------------------------------------------------------- cross-lingual view
    cross_rows: List[Dict] = []
    for model in sorted({r["model"] for r in ok_runs}):
        for lang in languages:
            subset = [r for r in ok_runs if r["model"] == model and r["doc_lang"] == lang]
            if not subset:
                continue
            quality = [r["overall_quality"] for r in subset if r.get("overall_quality") is not None]
            faithful = [r["factual_consistency"] for r in subset if r.get("factual_consistency") is not None]
            cross_rows.append({
                "model": model, "model_name": model_names.get(model, model),
                "trained_for": subset[0].get("trained_for"),
                "doc_lang": lang, "language_match": subset[0].get("language_match"),
                "n": len(subset),
                "quality_mean": bootstrap_mean_ci(quality)["mean"] if quality else None,
                "faithful_mean": bootstrap_mean_ci(faithful)["mean"] if faithful else None,
                "mean_summary_words": round(sum(r["summary_words"] for r in subset) / len(subset), 1),
            })
    write_csv(out / "cross_lingual.csv", cross_rows,
              ["model", "model_name", "trained_for", "doc_lang", "language_match", "n",
               "quality_mean", "faithful_mean", "mean_summary_words"])

    # ----------------------------------------------------------------- cost
    cost_rows: List[Dict] = []
    for model in sorted({r["model"] for r in ok_runs}):
        subset = [r for r in ok_runs if r["model"] == model]
        api = sum(r.get("api_cost_usd") or 0 for r in subset)
        compute = sum(r.get("compute_cost_usd") or 0 for r in subset)
        wall = sum(r.get("wall_seconds") or 0 for r in subset)
        gpu = sum(r.get("gpu_seconds") or 0 for r in subset)
        cpu = sum(r.get("cpu_seconds") or 0 for r in subset)
        tokens_in = sum(r.get("input_tokens") or 0 for r in subset)
        tokens_out = sum(r.get("output_tokens") or 0 for r in subset)
        cost_rows.append({
            "model": model, "model_name": model_names.get(model, model),
            "model_type": subset[0].get("model_type"), "runs": len(subset),
            "api_cost_usd": round(api, 6), "compute_cost_usd": round(compute, 6),
            "total_cost_usd": round(api + compute, 6),
            "cost_per_doc_usd": round((api + compute) / len(subset), 6),
            "wall_seconds": round(wall, 1), "cpu_seconds": round(cpu, 1),
            "gpu_seconds": round(gpu, 1),
            "seconds_per_doc": round(wall / len(subset), 2),
            "input_tokens": tokens_in, "output_tokens": tokens_out,
        })
    write_csv(out / "cost.csv", cost_rows,
              ["model", "model_name", "model_type", "runs", "api_cost_usd", "compute_cost_usd",
               "total_cost_usd", "cost_per_doc_usd", "wall_seconds", "cpu_seconds", "gpu_seconds",
               "seconds_per_doc", "input_tokens", "output_tokens"])

    write_report(folder, out, manifest, runs, ok_runs, failures,
                 stats_by_language, cross_rows, cost_rows, model_names,
                 degenerate, completed)
    print(f"wrote {out}/report.md and 5 CSVs")


def write_report(folder, out, manifest, runs, ok_runs, failures,
                 stats_by_language, cross_rows, cost_rows, model_names,
                 degenerate=(), completed=()) -> None:
    lines: List[str] = []
    add = lines.append

    add(f"# Experiment report — {manifest.get('name', folder.name)}")
    add("")
    add(f"- Created: {manifest.get('created_at', 'unknown')}")
    add(f"- Documents: {manifest.get('documents', '?')} {manifest.get('documents_by_language', '')}")
    add(f"- Runs: {len(ok_runs)} successful, {len(failures)} failed")
    add(f"- Compression target: {manifest.get('ratio', '?')}")
    machine = manifest.get("machine", {})
    add(f"- Machine: {machine.get('platform', '?')} ({machine.get('cpu_count', '?')} cores); "
        f"prices as of {machine.get('prices_as_of', '?')}")
    add("")
    add("Every model was run on every document regardless of language. Rows labelled "
        "`cross_language` are models applied to a language they were not trained on — "
        "that is deliberate and is the point of the cross-lingual comparison.")
    add("")

    # ---- per language tables
    for lang, block in stats_by_language.items():
        n_docs = len(block["documents"])
        add(f"## {lang.upper()} documents (n = {n_docs})")
        add("")
        if block["incomplete"]:
            add(f"Excluded from the tables (did not complete every document): "
                f"{', '.join(block['incomplete'])}.")
            add("")
        header = "| Model | trained for | " + " | ".join(
            label + (" ↓" if d == "lower" else " ↑" if d == "higher" else "")
            for _, label, d in METRICS if _ in block["metrics"]
        ) + " |"
        add(header)
        add("|" + "---|" * (header.count("|") - 1))
        for model in block["models"]:
            trained = next((r.get("trained_for") for r in ok_runs if r["model"] == model), "?")
            cells = []
            for metric, _, direction in METRICS:
                stats = block["metrics"].get(metric)
                if not stats:
                    continue
                entry = stats["per_system"].get(model)
                if not entry or entry["mean"] is None:
                    cells.append("—")
                    continue
                value = (num(entry["mean"], 2) if metric in PLAIN_NUMBER_METRICS
                         else pct(entry["mean"]))
                cells.append(value)
            add(f"| {model_names.get(model, model)} | {trained} | " + " | ".join(cells) + " |")
        add("")

        # significance, only where the document count supports it
        headline = block["metrics"].get("overall_quality", {})
        if headline.get("pairwise"):
            if headline.get("significance_tested"):
                add("### Pairwise comparisons — composite quality")
                add("")
                add("| Comparison | Difference | p (Holm) | Significant | Cohen's d |")
                add("|---|---|---|---|---|")
                for pair, result in headline["pairwise"].items():
                    add(f"| {model_names.get(result['system_a'], result['system_a'])} vs "
                        f"{model_names.get(result['system_b'], result['system_b'])} "
                        f"| {pct(result['mean_difference'])} | {result.get('p_adjusted', '—')} "
                        f"| {'yes' if result.get('significant') else 'no'} | {result.get('cohens_d', '—')} |")
                add("")
            else:
                add(f"_No p-values: n = {n_docs} documents, below the minimum of "
                    f"{MIN_N_FOR_SIGNIFICANCE} for a meaningful paired test. Differences above are "
                    f"descriptive._")
                add("")

    # ---- cross-lingual
    add("## Cross-lingual behaviour")
    add("")
    add("| Model | trained for | document language | match | n | composite | faithful | summary words |")
    add("|---|---|---|---|---|---|---|---|")
    for row in cross_rows:
        add(f"| {row['model_name']} | {row['trained_for']} | {row['doc_lang']} | {row['language_match']} "
            f"| {row['n']} | {pct(row['quality_mean'])} | {pct(row['faithful_mean'])} "
            f"| {row['mean_summary_words']} |")
    add("")
    # in-language vs cross-language deltas, where both exist for one model
    by_model: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for row in cross_rows:
        by_model[row["model"]][row["language_match"]] = row
    deltas = []
    for model, matches in by_model.items():
        if "in_language" in matches and "cross_language" in matches:
            a, b = matches["in_language"], matches["cross_language"]
            if a["quality_mean"] is not None and b["quality_mean"] is not None:
                deltas.append((model, a, b, a["quality_mean"] - b["quality_mean"]))
    if deltas:
        add("### In-language vs. cross-language")
        add("")
        add("| Model | in-language | cross-language | drop |")
        add("|---|---|---|---|")
        for model, a, b, delta in sorted(deltas, key=lambda x: -x[3]):
            add(f"| {a['model_name']} | {pct(a['quality_mean'])} ({a['doc_lang']}) "
                f"| {pct(b['quality_mean'])} ({b['doc_lang']}) | {pct(delta)} |")
        add("")

    # ---- cost
    add("## Cost and compute")
    add("")
    add("API cost is real, from provider token counts. Compute cost is notional: "
        "measured time priced at a configured hourly rate, so local and paid systems "
        "sit on one axis. Both are per this machine and this run.")
    add("")
    add("| Model | type | runs | API $ | compute $ (notional) | $/doc | s/doc | tokens in/out |")
    add("|---|---|---|---|---|---|---|---|")
    for row in sorted(cost_rows, key=lambda r: -r["total_cost_usd"]):
        tokens = f"{row['input_tokens']}/{row['output_tokens']}" if row["input_tokens"] else "—"
        add(f"| {row['model_name']} | {row['model_type']} | {row['runs']} | "
            f"${row['api_cost_usd']:.4f} | ${row['compute_cost_usd']:.4f} | "
            f"${row['cost_per_doc_usd']:.5f} | {row['seconds_per_doc']:.1f} | {tokens} |")
    total_api = sum(r["api_cost_usd"] for r in cost_rows)
    total_wall = sum(r["wall_seconds"] for r in cost_rows)
    add("")
    add(f"**Total real API spend: ${total_api:.4f}. Total wall time: {total_wall / 60:.1f} minutes "
        f"({total_wall / 3600:.1f} hours).**")
    add("")

    # ---- why the run took as long as it did -------------------------------
    add("### Where the time went")
    add("")
    by_model_runs = defaultdict(list)
    for run in ok_runs:
        by_model_runs[run["model"]].append(run)
    slowest = sorted(cost_rows, key=lambda r: -r["seconds_per_doc"])
    if slowest and total_wall > 0:
        top = slowest[0]
        share = 100 * top["wall_seconds"] / total_wall
        add(f"The run took **{total_wall / 3600:.1f} hours** of wall-clock time. "
            f"It was dominated by **{top['model_name']}** at {top['seconds_per_doc']:.1f} s/document, "
            f"which alone accounts for {share:.0f}% of total compute.")
        add("")
    add("| Model | s/doc | mean chunks/doc | context window | share of total time |")
    add("|---|---|---|---|---|")
    CONTEXT = {
        "extractive": "unbounded", "distilbart": "1,024", "bart-large": "1,024",
        "pegasus": "1,024", "mt5-xlsum": "512", "led-large-book": "16,384",
        "flan-t5-large": "1,024", "turkish-mbart": "1,024",
        # Qwen supports 8,192 but was run at 3,072 to fit the T4 (METHODOLOGY 1a).
        "qwen-llm": "3,072",
        "gpt-4": "8,192", "gpt-4o-mini": "128,000",
    }
    for row in slowest:
        runs_for_model = by_model_runs[row["model"]]
        chunks = [r["chunks"] for r in runs_for_model if r.get("chunks")]
        mean_chunks = f"{sum(chunks) / len(chunks):.1f}" if chunks else "—"
        share = 100 * row["wall_seconds"] / total_wall if total_wall else 0
        add(f"| {row['model_name']} | {row['seconds_per_doc']:.1f} | {mean_chunks} "
            f"| {CONTEXT.get(row['model'], '—')} | {share:.0f}% |")
    add("")
    add("Runtime is driven by **context window**, not parameter count: a model whose window is "
        "shorter than the document must summarize it in several chunk passes and then summarize "
        "the result, so cost scales with the number of chunks rather than with model size. "
        "This is a measured efficiency finding, not an implementation artefact — the same "
        "hierarchical procedure is applied to every system whose window is exceeded.")
    add("")

    # ---- silent failures: completed, but produced nothing
    if degenerate:
        add("## Silent failures (completed, produced no summary)")
        add("")
        add("These runs returned without error, so the runner recorded them as successful, but the "
            "summary is empty. They are **excluded from every metric, interval and significance "
            "test above**, and the documents they occurred on are excluded for all systems so that "
            "paired comparisons still rest on one identical document set. They are listed here "
            "because a silent-failure rate is a property of the system, not an accident of the run.")
        add("")
        add("An empty summary is not scored as bad by a reference-free suite — it is scored as "
            "*perfect* on every measure that asks whether anything is wrong: faithfulness 100% "
            "(nothing to contradict), lexical cohesion 100% (nothing incoherent), redundancy 0% "
            "(nothing repeated), composing to 25.0%. Left in the means, these runs **raise** the "
            "score of the system that failed.")
        add("")
        by_model_lang: Dict[tuple, int] = defaultdict(int)
        attempted: Dict[tuple, int] = defaultdict(int)
        for r in completed:
            attempted[(r["model"], r["doc_lang"])] += 1
            if is_degenerate(r):
                by_model_lang[(r["model"], r["doc_lang"])] += 1
        add("| Model | Language | Empty | Attempted | Silent-failure rate |")
        add("|---|---|---|---|---|")
        for key in sorted(by_model_lang, key=lambda k: (-by_model_lang[k], k)):
            n = by_model_lang[key]
            add(f"| {model_names.get(key[0], key[0])} | {key[1]} | {n} | {attempted[key]} | "
                f"**{n / attempted[key] * 100:.0f}%** |")
        add("")
        dropped = sorted({r["doc_id"] for r in degenerate})
        add(f"Documents excluded from all comparisons as a result: **{len(dropped)}** "
            f"({', '.join(dropped[:6])}{', …' if len(dropped) > 6 else ''}).")
        add("")

    # ---- failures
    if failures:
        add("## Failures")
        add("")
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for failure in failures:
            grouped[failure.get("model", "?")].append(failure)
        for model, items in grouped.items():
            example = items[0].get("error", "")[:200]
            add(f"- **{model_names.get(model, model)}**: {len(items)} failed — e.g. `{example}`")
        add("")

    add("## How to read the summaries")
    add("")
    add(f"Each document has a folder under `{folder}/docs/<doc_id>/` containing the source text, "
        f"the reference summary, and one file per model with a header of its scores. That is the "
        f"place to check whether a low score means lost context, lost meaning, or gibberish — the "
        f"metrics say how much, the text says what kind.")
    add("")

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="experiment folder, e.g. experiments/pilot")
    aggregate(Path(parser.parse_args().folder))


if __name__ == "__main__":
    main()
