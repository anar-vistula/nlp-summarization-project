#!/usr/bin/env python3
"""
Classify the OUTPUT language of every stored summary and report drift.

Why this exists as a script rather than an ad-hoc analysis: findings B6 and B7
rest on these counts, and the first pass at them used `langdetect` alone, which
misclassified Chinese summaries containing Latin-script Turkish proper nouns
("Kırıkkalespor", "Şebinkarahisar Madeni") as Turkish or English. Script
evidence is decisive where langdetect is probabilistic, so script is checked
first and langdetect is consulted only to separate languages sharing the Latin
alphabet.

Usage:
    python scripts/output_language.py experiments/main_run [--csv out.csv]
"""
import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langdetect import DetectorFactory, detect  # noqa: E402

DetectorFactory.seed = 0

CJK_RE = re.compile("[぀-ヿ㐀-䶿一-鿿豈-﫿]")
CYRILLIC_RE = re.compile("[Ѐ-ӿ]")
ARABIC_RE = re.compile("[؀-ۿ]")
TR_CHARS = set("çğışöüÇĞİŞÖÜ")

# A summary is called a script match when that script carries at least this
# share of its characters; well below any real text, well above stray quoting.
SCRIPT_SHARE = 0.05


def classify(text: str) -> str:
    """Return the output language/script label for one summary."""
    if not text.strip():
        return "empty"
    n = max(1, len(text))
    for label, pattern in (("zh", CJK_RE), ("ru", CYRILLIC_RE), ("ar", ARABIC_RE)):
        if len(pattern.findall(text)) / n >= SCRIPT_SHARE:
            return label
    # Latin-script: Turkish diacritics are strong evidence; langdetect decides
    # the rest. Both are needed — short Turkish summaries can lack diacritics.
    tr_share = sum(c in TR_CHARS for c in text) / n
    try:
        guess = detect(text)
    except Exception:
        guess = "unknown"
    if tr_share >= 0.01:
        return "tr"
    return guess if guess in ("tr", "en") else f"other:{guess}"


def read_summary(path: Path) -> str:
    return "\n".join(
        l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith("#")
    ).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    runs = [json.loads(l) for l in (args.folder / "runs.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for r in runs:
        if not r.get("ok"):
            continue
        f = args.folder / "docs" / r["doc_id"] / f"{r['model']}.txt"
        if not f.exists():
            continue
        out_lang = classify(read_summary(f))
        rows.append({
            "model": r["model"], "doc_id": r["doc_id"], "doc_lang": r["doc_lang"],
            "out_lang": out_lang, "matches": out_lang == r["doc_lang"],
            "chunks": r.get("chunks"), "doc_words": r.get("doc_words"),
            "rouge_l_f1": r.get("rouge_l_f1"), "overall_quality": r.get("overall_quality"),
            "factual_consistency": r.get("factual_consistency"),
            "abstractiveness": r.get("abstractiveness"), "summary_words": r.get("summary_words"),
        })

    # ---- per model x document language: where does the output end up?
    print("Output language of every stored summary\n")
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[(row["model"], row["doc_lang"])][row["out_lang"]] += 1
    print(f"{'model':16} {'input':5} {'n':>4}  output language distribution")
    for key in sorted(grouped):
        c = grouped[key]
        dist = "  ".join(f"{k}={v}" for k, v in c.most_common())
        print(f"{key[0]:16} {key[1]:5} {sum(c.values()):4}  {dist}")

    # ---- metrics conditioned on output language, where it varies
    print("\nMetrics conditioned on output language (only where a model drifts)\n")
    for key in sorted(grouped):
        if len(grouped[key]) < 2:
            continue
        subset = [r for r in rows if (r["model"], r["doc_lang"]) == key]
        print(f"{key[0]} on {key[1]} input (n={len(subset)}):")
        print(f"  {'output':8} {'n':>4} {'ROUGE-L':>8} {'quality':>8} {'faith':>7} {'novel':>7} {'words':>6} {'chunks':>7} {'docwords':>9}")
        for lang in sorted({r["out_lang"] for r in subset}):
            g = [r for r in subset if r["out_lang"] == lang]
            def m(f):
                v = [r[f] for r in g if r[f] is not None]
                return statistics.mean(v) if v else float("nan")
            print(f"  {lang:8} {len(g):4} {m('rouge_l_f1')*100:7.1f}% {m('overall_quality')*100:7.1f}% "
                  f"{m('factual_consistency')*100:6.1f}% {m('abstractiveness')*100:6.1f}% "
                  f"{m('summary_words'):6.0f} {m('chunks'):7.2f} {m('doc_words'):9.0f}")
        print()

    # ---- drift against chunk count, the B7 mechanism
    print("Language retention by chunk count (models that drift)\n")
    for key in sorted(grouped):
        if len(grouped[key]) < 2:
            continue
        subset = [r for r in rows if (r["model"], r["doc_lang"]) == key and r["chunks"]]
        if not subset:
            continue
        buckets = defaultdict(lambda: [0, 0])
        for r in subset:
            b = min(int(r["chunks"]), 5)
            buckets[b][0] += 1
            buckets[b][1] += int(r["matches"])
        print(f"{key[0]} on {key[1]} input:")
        for b in sorted(buckets):
            n, k = buckets[b]
            print(f"  {b}{'+' if b == 5 else ' '} chunks: n={n:3}  stayed in-language {k:3}/{n:<3} = {k/n*100:4.0f}%")
        print()

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
