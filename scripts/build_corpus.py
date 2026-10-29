#!/usr/bin/env python
"""
Build the evaluation corpus.

Writes one JSONL file per language to corpus/, where each line is a document
with a human-written reference summary and full provenance. Documents are the
unit of statistical analysis, so the corpus is built once, snapshotted into
each experiment, and never silently regenerated.

Sources
-------
  arxiv    ccdv/arxiv-summarization    scientific papers, abstract as reference
  pubmed   ccdv/pubmed-summarization   biomedical papers, abstract as reference
  trwiki   Turkish Wikipedia           featured/good articles, lead as reference

On the Turkish source: there is no public Turkish long-document summarization
dataset (TR-News averages 341 words, wikipedia-tr-summarization 277). Turkish
Wikipedia's featured ("Seçkin maddeler") and good ("Kaliteli maddeler")
articles are 4,000-8,500 words of reviewed prose, and their lead section is a
human-written standalone summary — the same convention WikiSum uses. The lead
is descriptive rather than a true abstract, which belongs in the limitations.

Examples
--------
    python scripts/build_corpus.py --source arxiv  --n 50 --out corpus/en.jsonl
    python scripts/build_corpus.py --source pubmed --n 50 --out corpus/en.jsonl --append
    python scripts/build_corpus.py --source trwiki --n 100 --out corpus/tr.jsonl
    python scripts/build_corpus.py --stats corpus/en.jsonl
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "diploma-project-corpus/1.0 (academic thesis; long-document summarization study)"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
WIKI_API = "https://tr.wikipedia.org/w/api.php"

# Wikipedia rate-limits anonymous API clients hard (observed 429s at ~1 req/s),
# so requests are spaced and retried with backoff rather than hammered.
WIKI_DELAY_SECONDS = 2.5
MAX_RETRIES = 5


# --------------------------------------------------------------------------- #
# HTTP with backoff
# --------------------------------------------------------------------------- #

def fetch_json(url: str, delay: float = 0.0) -> Dict:
    """GET JSON, retrying on 429/5xx with exponential backoff."""
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        if delay:
            time.sleep(delay)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code in (429, 500, 502, 503, 504):
                wait = float(error.headers.get("Retry-After") or 0) or (2 ** attempt) * 3
                print(f"    HTTP {error.code}; waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except Exception as error:           # transient network problems
            last_error = error
            time.sleep((2 ** attempt) * 2)
    raise RuntimeError(f"giving up after {MAX_RETRIES} attempts: {last_error}")


def wiki_api(params: Dict) -> Dict:
    return fetch_json(
        f"{WIKI_API}?" + urllib.parse.urlencode({**params, "format": "json"}),
        delay=WIKI_DELAY_SECONDS,
    )


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

HF_SOURCES = {
    "arxiv": {
        "dataset": "ccdv/arxiv-summarization",
        "config": "section",
        "split": "train",
        "text_field": "article",
        "reference_field": "abstract",
        "lang": "en",
        "rows": 203_037,
        "license": "arXiv terms; distributed as ccdv/arxiv-summarization (Cohan et al., 2018)",
    },
    "pubmed": {
        "dataset": "ccdv/pubmed-summarization",
        "config": "section",
        "split": "train",
        "text_field": "article",
        "reference_field": "abstract",
        "lang": "en",
        "rows": 119_924,
        "license": "PubMed open access; distributed as ccdv/pubmed-summarization (Cohan et al., 2018)",
    },
}


def iter_hf_documents(source: str, seed: int, page_size: int = 50) -> Iterator[Dict]:
    """
    Stream documents from a HuggingFace dataset via the rows API.

    Offsets are sampled across the whole split (seeded, so a rebuild reproduces
    the same corpus) instead of reading from row 0, which would bias the corpus
    towards whatever happens to sit at the head of the file.
    """
    spec = HF_SOURCES[source]
    rng = random.Random(seed)
    offsets = list(range(0, max(spec["rows"] - page_size, 1), page_size))
    rng.shuffle(offsets)

    for offset in offsets:
        url = (
            f"{HF_ROWS}?dataset={urllib.parse.quote(spec['dataset'])}"
            f"&config={spec['config']}&split={spec['split']}"
            f"&offset={offset}&length={page_size}"
        )
        try:
            rows = fetch_json(url, delay=0.4).get("rows", [])
        except Exception as error:
            print(f"    page at offset {offset} failed: {error}", file=sys.stderr)
            continue
        for row in rows:
            record = row["row"]
            text = (record.get(spec["text_field"]) or "").strip()
            reference = (record.get(spec["reference_field"]) or "").strip()
            if not text or not reference:
                continue
            yield {
                "lang": spec["lang"],
                "source": source,
                "source_ref": f"{spec['dataset']}#{spec['split']}[{offset + row['row_idx'] % page_size}]",
                "license": spec["license"],
                "title": f"{source} document {row['row_idx']}",
                "text": text,
                "reference": reference,
            }


def iter_wikipedia_documents(seed: int) -> Iterator[Dict]:
    """
    Turkish Wikipedia featured and good articles, lead section as reference.

    Candidate titles come from the review categories via search; the article
    body is then fetched one page per request (the extracts API only returns
    full-article text for a single page at a time) and the lead separately.
    """
    rng = random.Random(seed)
    titles: List[str] = []
    for category in ("Seçkin maddeler", "Kaliteli maddeler"):
        offset = 0
        while offset < 500:
            payload = wiki_api({
                "action": "query", "list": "search",
                "srsearch": f'incategory:"{category}"',
                "srlimit": 50, "sroffset": offset, "srnamespace": 0,
            })
            hits = payload.get("query", {}).get("search", [])
            if not hits:
                break
            titles.extend(hit["title"] for hit in hits)
            if "continue" not in payload:
                break
            offset += 50
    # Drop list/season/discography pages: they are tables, not prose.
    noise = ("listesi", "Listesi", "listeleri", "sezonu", "Sezonu", "filmografi", "diskografi")
    titles = [t for t in dict.fromkeys(titles) if not any(n in t for n in noise)]
    rng.shuffle(titles)
    print(f"  {len(titles)} candidate Turkish articles", file=sys.stderr)

    for title in titles:
        try:
            body_payload = wiki_api({
                "action": "query", "prop": "extracts", "explaintext": 1, "titles": title,
            })
            page = next(iter(body_payload["query"]["pages"].values()))
            text = (page.get("extract") or "").strip()
            if not text:
                continue
            lead_payload = wiki_api({
                "action": "query", "prop": "extracts", "explaintext": 1,
                "exintro": 1, "titles": title,
            })
            lead_page = next(iter(lead_payload["query"]["pages"].values()))
            reference = (lead_page.get("extract") or "").strip()
        except Exception as error:
            print(f"    {title}: {error}", file=sys.stderr)
            continue
        if not reference:
            continue
        # The lead is part of the body; remove it so the model cannot simply
        # copy the reference out of the source text.
        body = text[len(reference):].strip() if text.startswith(reference) else text
        yield {
            "lang": "tr",
            "source": "trwiki",
            "source_ref": f"https://tr.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "license": "CC BY-SA 4.0",
            "title": title,
            "text": body,
            "reference": reference,
        }


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def build(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Dict] = {}
    if out_path.exists() and args.append:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["id"]] = record
        print(f"  {len(existing)} documents already in {out_path}", file=sys.stderr)

    seen_texts = {record["text"][:300] for record in existing.values()}
    source_counter = sum(1 for r in existing.values() if r["source"] == args.source)

    stream = (
        iter_wikipedia_documents(args.seed)
        if args.source == "trwiki"
        else iter_hf_documents(args.source, args.seed)
    )

    kept = 0
    inspected = 0
    for document in stream:
        if kept >= args.n:
            break
        inspected += 1
        words = len(document["text"].split())
        reference_words = len(document["reference"].split())
        if not (args.min_words <= words <= args.max_words):
            continue
        if reference_words < args.min_reference_words:
            continue
        if document["text"][:300] in seen_texts:
            continue
        seen_texts.add(document["text"][:300])

        source_counter += 1
        document.update({
            "id": f"{document['lang']}_{document['source']}_{source_counter:04d}",
            "words": words,
            "reference_words": reference_words,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        existing[document["id"]] = document
        kept += 1
        if kept % 10 == 0 or args.source == "trwiki":
            print(f"    kept {kept}/{args.n} ({inspected} inspected): "
                  f"{document['title'][:44]} — {words}w", file=sys.stderr)

    with out_path.open("w", encoding="utf-8") as handle:
        for record in existing.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  wrote {len(existing)} documents to {out_path} (+{kept} new from {args.source})")


def show_stats(path: str) -> None:
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print("empty corpus")
        return
    by_source: Dict[str, List[Dict]] = {}
    for record in records:
        by_source.setdefault(record["source"], []).append(record)
    print(f"{path}: {len(records)} documents")
    for source, group in sorted(by_source.items()):
        words = sorted(r["words"] for r in group)
        refs = sorted(r["reference_words"] for r in group)
        print(f"  {source:8} n={len(group):4}  words min/median/max = "
              f"{words[0]}/{words[len(words)//2]}/{words[-1]}  "
              f"(~{words[len(words)//2]/500:.1f} pages)  reference median {refs[len(refs)//2]}w")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["arxiv", "pubmed", "trwiki"])
    parser.add_argument("--n", type=int, default=50, help="documents to collect")
    parser.add_argument("--out", default="corpus/en.jsonl")
    parser.add_argument("--append", action="store_true", default=True,
                        help="merge into an existing file (default)")
    parser.add_argument("--fresh", dest="append", action="store_false",
                        help="overwrite the output file instead of merging")
    parser.add_argument("--min-words", type=int, default=1200,
                        help="lower bound: roughly 2.5 pages")
    parser.add_argument("--max-words", type=int, default=6000,
                        help="upper bound; keeps runs and API cost bounded")
    parser.add_argument("--min-reference-words", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--stats", metavar="PATH", help="print statistics for a corpus file and exit")
    args = parser.parse_args()

    if args.stats:
        show_stats(args.stats)
        return
    if not args.source:
        parser.error("--source is required (or use --stats)")
    build(args)


if __name__ == "__main__":
    main()
