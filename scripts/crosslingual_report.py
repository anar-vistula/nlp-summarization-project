#!/usr/bin/env python
"""
Cross-lingual analysis: how each system behaves on a language it was not
trained for, and which failure mode it exhibits.

Beyond the standard metrics, this computes two diagnostics that the main metric
set misses, both discovered by reading outputs rather than scores:

  script_fidelity   share of language-specific characters in the output
                    relative to the source. Catches a tokenizer that silently
                    deletes characters it has no vocabulary for (Pegasus drops
                    every Turkish ç/ğ/ı/ö/ş/ü), which the abstractiveness
                    metric misreads as paraphrase.

  checkable_density numbers and proper nouns per 100 words. A faithfulness
                    score is only meaningful when there is something to check;
                    degenerate output contains almost nothing, and therefore
                    scores as faithful.

    python scripts/crosslingual_report.py experiments/main_run
"""

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluation.text_utils import extract_capitalized_phrases, extract_numbers  # noqa: E402

# Characters that exist in the language but not in plain ASCII English.
LANGUAGE_CHARS = {
    "tr": set("çğıöşüÇĞİÖŞÜ"),
    "en": set(),
}


def script_fidelity(text: str, lang: str) -> Optional[float]:
    """Share of letters that are language-specific. None when not applicable."""
    chars = LANGUAGE_CHARS.get(lang) or set()
    if not chars:
        return None
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c in chars) / len(letters)


def checkable_density(text: str, lang: str) -> float:
    """Numbers plus proper-noun phrases per 100 words."""
    words = max(1, len(text.split()))
    items = len(extract_numbers(text)) + len(extract_capitalized_phrases(text, lang))
    return 100.0 * items / words


def read_summary(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ).strip()


def analyse(folder: Path) -> None:
    runs = [json.loads(l) for l in (folder / "runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    documents = {
        d["id"]: d for d in
        (json.loads(l) for l in (folder / "corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }

    cells: Dict[tuple, List[Dict]] = defaultdict(list)
    for record in runs:
        if not record.get("ok"):
            continue
        cells[(record["model"], record["doc_lang"])].append(record)

    # Source-side baselines per language.
    source_fidelity = {}
    for lang in {d["lang"] for d in documents.values()}:
        texts = [d["text"] for d in documents.values() if d["lang"] == lang]
        values = [script_fidelity(t, lang) for t in texts]
        values = [v for v in values if v is not None]
        source_fidelity[lang] = st.mean(values) if values else None

    print(f"\nCross-lingual analysis — {folder.name}\n")
    header = (f"{'model':18}{'lang':>5}{'match':>15}{'n':>5}{'quality':>9}{'rougeL':>8}"
              f"{'faithful':>10}{'novel':>8}{'script':>8}{'check/100w':>12}")
    print(header)
    print("-" * len(header))

    for (model, lang), records in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        fidelities, densities = [], []
        for record in records:
            summary = read_summary(folder / "docs" / record["doc_id"] / f"{model}.txt")
            if not summary:
                continue
            value = script_fidelity(summary, lang)
            if value is not None:
                fidelities.append(value)
            densities.append(checkable_density(summary, lang))

        def mean_of(key):
            values = [r[key] for r in records if r.get(key) is not None]
            return st.mean(values) if values else None

        def pct(value):
            return "—" if value is None else f"{value * 100:5.1f}%"

        script = "—"
        if fidelities:
            base = source_fidelity.get(lang)
            script = f"{st.mean(fidelities) * 100:5.2f}%"
            if base:
                script += f"/{base * 100:.1f}"
        print(f"{model:18}{lang:>5}{records[0].get('language_match', '?'):>15}{len(records):>5}"
              f"{pct(mean_of('overall_quality')):>9}{pct(mean_of('rouge_l_f1')):>8}"
              f"{pct(mean_of('factual_consistency')):>10}{pct(mean_of('abstractiveness')):>8}"
              f"{script:>8}{st.mean(densities) if densities else 0:>11.1f}")

    print("\nscript column: language-specific characters in output / in source (Turkish only).")
    print("A value near zero against a non-zero source means the tokenizer is deleting them,")
    print("which inflates 'novel' and destroys ROUGE without the content necessarily being wrong.")
    print("check/100w: numbers + proper nouns per 100 words. Faithfulness on a near-zero value")
    print("is uninformative — there was nothing to contradict.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    analyse(Path(parser.parse_args().folder))


if __name__ == "__main__":
    main()
