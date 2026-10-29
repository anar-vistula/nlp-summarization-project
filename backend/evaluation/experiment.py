"""
Experiment runner: evaluates every available model over a set of documents and
aggregates the results statistically.

Rationale. The per-request comparison endpoints score one document, which can
describe a system's behaviour but cannot establish that one system is better
than another. This runner performs the controlled experiment instead: the same
documents, the same compression target, every model, per-document scores, and
then means with bootstrap confidence intervals and pairwise significance tests.

Documents are grouped by language and never pooled across languages: the
English and the multilingual model sets are different, so a pooled mean would
compare systems on unequal document sets.
"""

import csv
import io
import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from backend.evaluation.metrics import SummaryEvaluator
from backend.evaluation.statistics import compare_systems
from backend.sample_texts import SAMPLES

# Scalar metrics that get aggregated across documents. Each entry maps a
# reporting name to a function pulling one number out of a metrics dict, plus
# whether higher is better ("direction") for ranking.
MetricAccessor = Callable[[Dict], Optional[float]]


def _get(path: List[str]) -> MetricAccessor:
    def accessor(metrics: Dict) -> Optional[float]:
        node = metrics
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node if isinstance(node, (int, float)) else None

    return accessor


AGGREGATED_METRICS: Dict[str, Dict] = {
    "overall_quality": {"accessor": _get(["overall_quality", "score"]), "direction": "higher"},
    "rouge_1_f1": {"accessor": _get(["rouge_scores", "rouge1", "f1"]), "direction": "higher"},
    "rouge_2_f1": {"accessor": _get(["rouge_scores", "rouge2", "f1"]), "direction": "higher"},
    "rouge_l_f1": {"accessor": _get(["rouge_scores", "rougeL", "f1"]), "direction": "higher"},
    "meteor": {"accessor": _get(["meteor_score", "score"]), "direction": "higher"},
    "information_density": {"accessor": _get(["information_density", "score"]), "direction": "higher"},
    "topic_coverage": {"accessor": _get(["topic_coverage", "score"]), "direction": "higher"},
    "source_coverage": {"accessor": _get(["coverage_score", "score"]), "direction": "higher"},
    "lexical_cohesion": {"accessor": _get(["coherence_score", "score"]), "direction": "higher"},
    "redundancy": {"accessor": _get(["redundancy_score", "score"]), "direction": "lower"},
    "factual_consistency": {"accessor": _get(["factual_consistency", "score"]), "direction": "higher"},
    "abstractiveness": {"accessor": _get(["abstractiveness", "novel_1gram_ratio"]), "direction": "descriptive"},
    "extractive_density": {"accessor": _get(["extractive_fragments", "density"]), "direction": "descriptive"},
    "summary_words": {"accessor": _get(["compression_ratio", "summary_words"]), "direction": "descriptive"},
    "compression_ratio": {"accessor": _get(["compression_ratio", "ratio"]), "direction": "lower"},
    "runtime_seconds": {"accessor": lambda m: m.get("_runtime_seconds"), "direction": "lower"},
}

# Metrics the pairwise significance tests are run on. Testing every metric
# would multiply the comparison family and cost power for no reason; these are
# the headline claims a thesis makes.
TESTED_METRICS = [
    "overall_quality",
    "rouge_l_f1",
    "information_density",
    "factual_consistency",
]


# Completed runs are written here so that a long experiment survives a server
# restart (the dev server reloads on every code edit) and can be cited later.
RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiment_results"


class ExperimentRunner:
    """Runs the multi-document, multi-model evaluation and aggregates it."""

    def __init__(self, comparison_service, evaluator: Optional[SummaryEvaluator] = None):
        self.comparison = comparison_service
        self.evaluator = evaluator or SummaryEvaluator()
        self.last_run: Optional[Dict] = None

    # ------------------------------------------------------------------ #
    # Document set
    # ------------------------------------------------------------------ #

    @staticmethod
    def list_documents() -> List[Dict]:
        """Built-in documents available as experiment material."""
        return [
            {
                "key": key,
                "title": sample["title"],
                "language": sample["language"],
                "category": sample["category"],
                "word_count": len(sample["text"].split()),
                "has_reference": bool(sample.get("reference_summary")),
            }
            for key, sample in SAMPLES.items()
        ]

    def _resolve_documents(
        self,
        document_keys: Optional[List[str]],
        extra_documents: Optional[List[Dict]],
    ) -> List[Dict]:
        """Build the document list: built-in samples plus any supplied text."""
        documents: List[Dict] = []
        keys = document_keys if document_keys is not None else list(SAMPLES.keys())
        for key in keys:
            sample = SAMPLES.get(key)
            if not sample:
                continue
            documents.append(
                {
                    "key": key,
                    "title": sample["title"],
                    "language": sample["language"],
                    "text": sample["text"],
                    "reference_summary": sample.get("reference_summary"),
                }
            )
        for i, doc in enumerate(extra_documents or []):
            text = (doc.get("text") or "").strip()
            if not text:
                continue
            documents.append(
                {
                    "key": doc.get("key") or f"custom_{i + 1}",
                    "title": doc.get("title") or f"Custom document {i + 1}",
                    "language": doc.get("language")
                    or self.evaluator.preprocessor.detect_language(text),
                    "text": text,
                    "reference_summary": doc.get("reference_summary"),
                }
            )
        return documents

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    def run(
        self,
        document_keys: Optional[List[str]] = None,
        model_keys: Optional[List[str]] = None,
        ratio: float = 0.3,
        extra_documents: Optional[List[Dict]] = None,
    ) -> Iterator[Dict]:
        """
        Run the experiment, yielding progress events.

        Events: `plan`, `result` (one model on one document), `error`,
        `aggregate` (the statistics), `done`. The caller streams these to the
        UI; `run_sync` collects them instead.
        """
        documents = self._resolve_documents(document_keys, extra_documents)
        if not documents:
            yield {"stage": "error", "data": {"message": "No documents selected."}}
            return

        # Plan: which models apply to each document's language.
        models_by_lang: Dict[str, List[Dict]] = {}
        for doc in documents:
            lang = doc["language"]
            if lang not in models_by_lang:
                available = self.comparison.get_available_models(lang)
                if model_keys:
                    available = [m for m in available if m["key"] in model_keys]
                models_by_lang[lang] = available

        total_runs = sum(len(models_by_lang[d["language"]]) for d in documents)
        yield {
            "stage": "plan",
            "data": {
                "documents": [
                    {
                        "key": d["key"],
                        "title": d["title"],
                        "language": d["language"],
                        "word_count": len(d["text"].split()),
                        "has_reference": bool(d.get("reference_summary")),
                    }
                    for d in documents
                ],
                "models_by_language": {
                    lang: [{"key": m["key"], "name": m["name"], "type": m["type"]} for m in models]
                    for lang, models in models_by_lang.items()
                },
                "total_runs": total_runs,
                "ratio": ratio,
            },
        }

        results: Dict[str, Dict[str, Dict]] = {}
        errors: List[Dict] = []
        completed = 0

        for doc in documents:
            lang = doc["language"]
            results[doc["key"]] = {}
            for model_info in models_by_lang[lang]:
                key = model_info["key"]
                started = time.perf_counter()
                try:
                    produced = self.comparison.run_model(model_info, doc["text"], ratio, lang)
                    runtime = time.perf_counter() - started
                    metrics = self.evaluator.evaluate(
                        doc["text"],
                        produced["summary"],
                        reference_summary=doc.get("reference_summary"),
                        lang=lang,
                    )
                    metrics["_runtime_seconds"] = round(runtime, 3)
                    results[doc["key"]][key] = {
                        "model_name": model_info["name"],
                        "model_type": model_info["type"],
                        "summary": produced["summary"],
                        "runtime_seconds": round(runtime, 3),
                        "metrics": metrics,
                    }
                    completed += 1
                    yield {
                        "stage": "result",
                        "data": {
                            "document": doc["key"],
                            "document_title": doc["title"],
                            "language": lang,
                            "model": key,
                            "model_name": model_info["name"],
                            "runtime_seconds": round(runtime, 3),
                            "summary": produced["summary"],
                            "scores": {
                                name: spec["accessor"](metrics)
                                for name, spec in AGGREGATED_METRICS.items()
                            },
                            "progress": {"completed": completed, "total": total_runs},
                        },
                    }
                except Exception as exc:
                    completed += 1
                    errors.append(
                        {
                            "document": doc["key"],
                            "model": key,
                            "model_name": model_info["name"],
                            "error": str(exc),
                        }
                    )
                    yield {
                        "stage": "error",
                        "data": {
                            "document": doc["key"],
                            "model": key,
                            "model_name": model_info["name"],
                            "message": str(exc),
                            "progress": {"completed": completed, "total": total_runs},
                        },
                    }

        aggregate = self.aggregate(documents, results)
        self.last_run = {
            "documents": [
                {k: v for k, v in d.items() if k != "text"} for d in documents
            ],
            "results": results,
            "errors": errors,
            "aggregate": aggregate,
            "ratio": ratio,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        saved = self._save_run()
        yield {"stage": "aggregate", "data": aggregate}
        yield {"stage": "done", "data": {"errors": errors, "saved_to": saved}}

    def _save_run(self) -> Optional[str]:
        """Write the finished run to disk as JSON + CSV. Never fatal."""
        if not self.last_run:
            return None
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            (RESULTS_DIR / f"run_{stamp}.json").write_text(
                json.dumps(self.last_run, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            csv_path = RESULTS_DIR / f"run_{stamp}.csv"
            csv_path.write_text(self.last_run_csv(), encoding="utf-8")
            return str(csv_path)
        except OSError:
            return None

    def latest_saved_csv(self) -> Optional[str]:
        """Most recent run CSV from disk, for when the process has restarted."""
        if not RESULTS_DIR.is_dir():
            return None
        files = sorted(RESULTS_DIR.glob("run_*.csv"))
        if not files:
            return None
        try:
            return files[-1].read_text(encoding="utf-8")
        except OSError:
            return None

    def run_sync(self, **kwargs) -> Dict:
        """Run the experiment and return the final result (no streaming)."""
        for _ in self.run(**kwargs):
            pass
        return self.last_run or {}

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #

    def aggregate(self, documents: List[Dict], results: Dict[str, Dict[str, Dict]]) -> Dict:
        """
        Aggregate per-document scores per language.

        Only models that produced a summary for *every* document of a language
        enter the statistics for that language, so all systems being compared
        rest on the same document set. Models that failed somewhere are
        reported separately rather than compared on a subset.
        """
        by_language: Dict[str, Dict] = {}
        languages = sorted({d["language"] for d in documents})

        for lang in languages:
            docs = [d for d in documents if d["language"] == lang]
            doc_keys = [d["key"] for d in docs]
            model_keys = sorted({m for k in doc_keys for m in results.get(k, {})})

            complete, incomplete = [], []
            for model in model_keys:
                if all(model in results.get(k, {}) for k in doc_keys):
                    complete.append(model)
                else:
                    incomplete.append(model)

            model_names = {
                m: results[k][m]["model_name"]
                for k in doc_keys
                for m in results.get(k, {})
                if m in results.get(k, {})
            }

            metric_stats: Dict[str, Dict] = {}
            for metric_name, spec in AGGREGATED_METRICS.items():
                scores_by_model: Dict[str, List[float]] = {}
                for model in complete:
                    values = [
                        spec["accessor"](results[k][model]["metrics"]) for k in doc_keys
                    ]
                    if all(v is not None for v in values) and values:
                        scores_by_model[model] = [float(v) for v in values]
                if not scores_by_model:
                    continue
                stats = compare_systems(scores_by_model)
                stats["direction"] = spec["direction"]
                stats["tested"] = metric_name in TESTED_METRICS
                if metric_name not in TESTED_METRICS:
                    # Keep the table light: CIs for every metric, pairwise
                    # significance only for the pre-registered headline metrics.
                    stats["pairwise"] = {}
                metric_stats[metric_name] = stats

            by_language[lang] = {
                "documents": [
                    {"key": d["key"], "title": d["title"], "words": len(d["text"].split())}
                    for d in docs
                ],
                "n_documents": len(docs),
                "references_available": sum(1 for d in docs if d.get("reference_summary")),
                "models_compared": complete,
                "models_excluded": incomplete,
                "model_names": model_names,
                "metrics": metric_stats,
            }

        return {
            "by_language": by_language,
            "tested_metrics": TESTED_METRICS,
            "note": (
                "Languages are aggregated separately because the available model "
                "sets differ by language. Significance tests are pre-limited to the "
                "headline metrics to keep the multiple-comparison family small."
            ),
        }

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def last_run_csv(self) -> str:
        """
        Per-document, per-model scores of the last run as CSV — the raw data
        behind the tables, for the appendix or for re-analysis elsewhere.
        """
        if not self.last_run:
            return ""
        buffer = io.StringIO()
        columns = ["document", "title", "language", "model", "model_name", "has_reference"]
        columns += list(AGGREGATED_METRICS.keys())
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        doc_meta = {d["key"]: d for d in self.last_run["documents"]}
        for doc_key, per_model in self.last_run["results"].items():
            meta = doc_meta.get(doc_key, {})
            for model_key, entry in per_model.items():
                row = {
                    "document": doc_key,
                    "title": meta.get("title", ""),
                    "language": meta.get("language", ""),
                    "model": model_key,
                    "model_name": entry["model_name"],
                    "has_reference": bool(meta.get("reference_summary")),
                }
                for name, spec in AGGREGATED_METRICS.items():
                    value = spec["accessor"](entry["metrics"])
                    row[name] = "" if value is None else value
                writer.writerow(row)
        return buffer.getvalue()
