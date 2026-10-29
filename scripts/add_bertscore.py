#!/usr/bin/env python3
"""
Add BERTScore to every stored run, as a STANDALONE metric.

Why standalone. `REFERENCE_WEIGHTS` in metrics.py assigns bert_score_f1 a
weight of 0.20, so computing BERTScore inside SummaryEvaluator.evaluate()
would change the reference composite — and therefore overall_quality — for all
2,000 runs. That was rejected for two reasons:

  1. Comparability. Every number already reported comes from runs scored
     without it; silently redefining the composite would make the thesis's
     tables incomparable with its own intermediate results, and the composite
     has already been rescored once.
  2. It would defeat the purpose. BERTScore is in this study as a *contrast*
     to surface overlap — the A-section argues that surface metrics fail under
     distribution shift, and the interesting question is whether a semantic
     metric agrees with ROUGE or not. Averaging the two into one number
     destroys exactly the comparison it was added to make.

So this script writes bert_score_f1 / _precision / _recall as new fields and
leaves overall_quality untouched.

Models differ by language: bert-score selects roberta-large for English and
bert-base-multilingual-cased for Turkish. **BERTScore is therefore comparable
within a language and not across languages** — which matches how this study
aggregates anyway (never pooled).

Empty summaries get null rather than a score: BERTScore of an empty string
against a reference is not a measurement, and those runs are reported as a
silent-failure rate instead (see aggregate.py).

Usage:
    python scripts/add_bertscore.py experiments/main_run [--dry-run]
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_summary(path: Path) -> str:
    return "\n".join(
        l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith("#")
    ).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    runs_path = args.folder / "runs.jsonl"
    runs = [json.loads(l) for l in runs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = {}
    for line in (args.folder / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            docs[d["id"]] = d

    # Collect work, grouped by language so each language uses its own model.
    todo = defaultdict(list)       # lang -> [(index, candidate, reference)]
    empty = 0
    for i, r in enumerate(runs):
        if not r.get("ok"):
            continue
        f = args.folder / "docs" / r["doc_id"] / f"{r['model']}.txt"
        if not f.exists():
            continue
        summary = read_summary(f)
        reference = (docs.get(r["doc_id"]) or {}).get("reference")
        if not reference:
            continue
        if not summary:
            empty += 1
            runs[i]["bert_score_f1"] = None
            runs[i]["bert_score_precision"] = None
            runs[i]["bert_score_recall"] = None
            continue
        todo[r["doc_lang"]].append((i, summary, reference))

    print(f"to score: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(todo.items()))
          + f" | empty summaries left null: {empty}")

    from bert_score import score as bert_score

    for lang, items in sorted(todo.items()):
        t0 = time.time()
        cands = [c for _, c, _ in items]
        refs = [r for _, _, r in items]
        P, R, F = bert_score(cands, refs, lang=lang, verbose=False, batch_size=args.batch_size)
        for (i, _, _), p, r_, f in zip(items, P, R, F):
            runs[i]["bert_score_precision"] = round(float(p), 4)
            runs[i]["bert_score_recall"] = round(float(r_), 4)
            runs[i]["bert_score_f1"] = round(float(f), 4)
        print(f"  {lang}: {len(items)} pairs in {time.time()-t0:.0f}s "
              f"| mean F1 {statistics.mean(float(x) for x in F):.4f}")

    # Per model x language summary, so the result is readable before writing.
    print("\nmean BERTScore F1 by model and language:")
    grouped = defaultdict(list)
    for r in runs:
        if r.get("ok") and r.get("bert_score_f1") is not None:
            grouped[(r["model"], r["doc_lang"])].append(r["bert_score_f1"])
    for key in sorted(grouped):
        v = grouped[key]
        print(f"  {key[1]}  {key[0]:16} n={len(v):4}  F1={statistics.mean(v):.4f}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return
    shutil.copy2(runs_path, runs_path.with_suffix(".jsonl.prebert"))
    with runs_path.open("w", encoding="utf-8") as fh:
        for r in runs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwritten. previous runs.jsonl saved as {runs_path.name}.prebert")
    print("overall_quality was NOT changed — BERTScore is reported standalone.")


if __name__ == "__main__":
    main()
