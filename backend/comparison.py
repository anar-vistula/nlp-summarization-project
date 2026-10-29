"""
Multi-Model Comparison Service.
Runs the same text through ALL available models and compares results.
"""

import time
from typing import Optional, Dict, List
from backend.preprocessing import TextPreprocessor
from backend.summarizers.extractive import ExtractiveSummarizer
from backend.summarizers.abstractive import (
    AbstractiveSummarizer,
    MULTILINGUAL_MODELS,
    EN_MODELS,
    finetuned_turkish_available,
)
from backend.summarizers.paid_models import PaidModelSummarizer
from backend.evaluation.metrics import SummaryEvaluator


class ComparisonService:
    """
    Runs all available summarization models on the same text
    and produces a comprehensive comparison with metrics.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer()
        self.paid = PaidModelSummarizer()
        self.evaluator = SummaryEvaluator()

    def get_available_models(self, lang: str = "en", cross_lingual: bool = False) -> List[dict]:
        """
        Models offered for a document in `lang`.

        By default only models that suit the language are returned. With
        `cross_lingual=True` every runnable model is returned regardless of
        what it was trained on, so an English-only BART can be pointed at a
        Turkish document and vice versa. Those mismatched runs are the point of
        the cross-lingual comparison, not an error, and each entry carries a
        `language_match` label saying which regime it belongs to.
        """
        from backend.evaluation.model_registry import BY_KEY, language_match

        from backend.summarizers.paid_models import ColabGPUClient
        colab = ColabGPUClient()
        colab_models = colab.get_available_models() if colab.is_available() else []

        candidates: List[dict] = [
            {"key": "extractive", "name": "TextRank (Extractive)", "type": "extractive"},
            {"key": "distilbart", "name": "DistilBART-CNN", "type": "hf"},
            {"key": "bart-large", "name": "BART-Large-CNN", "type": "hf"},
            {"key": "pegasus", "name": "Pegasus-CNN/DailyMail", "type": "hf"},
            {"key": "mt5-xlsum", "name": "mT5-XLSum (Multilingual)", "type": "hf"},
        ]
        if finetuned_turkish_available():
            candidates.append({
                "key": "mt5-turkish-finetuned",
                "name": "mT5-Turkish (Fine-tuned)",
                "type": "hf",
            })

        # Anything the GPU server hosts. A model hosted there AND available
        # locally is served from the GPU, which keeps the laptop cool.
        listed = {m["key"] for m in candidates}
        for key in colab_models:
            entry = BY_KEY.get(key)
            if not entry:
                continue
            if key in listed:
                for model in candidates:
                    if model["key"] == key:
                        model["type"] = "colab"
            else:
                candidates.append({"key": key, "name": entry["name"], "type": "colab"})

        for model_key in self.paid.get_available_models():
            config = self.paid.SUPPORTED_MODELS[model_key]
            candidates.append({"key": model_key, "name": config["display_name"], "type": "paid"})

        models = []
        for model in candidates:
            match = language_match(model["key"], lang)
            # Language gate, unless the caller asked for the mismatches too.
            if not cross_lingual and match == "cross_language":
                continue
            models.append({**model, "language_match": match})
        return models

    def compare_all(
        self,
        text: str,
        reference_summary: Optional[str] = None,
        ratio: float = 0.3,
        models: Optional[List[str]] = None,
        cross_lingual: bool = False,
    ) -> Dict:
        """
        Run all (or selected) models on the same text and compare.

        Args:
            text: Input text to summarize
            reference_summary: Optional reference for ROUGE evaluation
            ratio: Target compression ratio
            models: List of model keys to use (None = all available)

        Returns:
            Comprehensive comparison results
        """
        preprocessed = self.preprocessor.preprocess(text)
        lang = preprocessed["language"]

        available = self.get_available_models(lang, cross_lingual=cross_lingual)
        if models:
            available = [m for m in available if m["key"] in models]

        results = {}
        errors = {}

        for model_info in available:
            key = model_info["key"]
            try:
                started = time.perf_counter()
                result = self.run_model(model_info, text, ratio, lang)
                runtime = time.perf_counter() - started

                results[key] = {
                    "name": model_info["name"],
                    "summary": result["summary"],
                    "compression_ratio": result.get("compression_ratio", 0),
                    "method": result.get("method", model_info["name"]),
                    "warning": result.get("warning"),
                    "reliability": result.get("reliability", "unknown"),
                    "runtime_seconds": round(runtime, 3),
                    "language_match": model_info.get("language_match"),
                }
            except Exception as e:
                errors[key] = {"name": model_info["name"], "error": str(e)}

        # Evaluate all summaries
        summaries_for_eval = {k: v["summary"] for k, v in results.items()}
        if summaries_for_eval:
            comparison = self.evaluator.compare_summaries(
                text, summaries_for_eval, reference_summary=reference_summary, lang=lang
            )
            # Attach metrics to each result
            for key in results:
                if key in comparison["individual_results"]:
                    results[key]["metrics"] = comparison["individual_results"][key]

            rankings = comparison["rankings"]
        else:
            rankings = {}

        return {
            "input_stats": {
                "word_count": preprocessed["word_count"],
                "sentence_count": preprocessed["num_sentences"],
                "language": lang,
            },
            "results": results,
            "errors": errors,
            "rankings": rankings,
            "models_used": len(results),
        }

    def run_model(self, model_info: Dict, text: str, ratio: float, lang: str) -> Dict:
        """
        Run one model on one text. Single dispatch point shared by the
        single-document comparison and the experiment runner, so both measure
        exactly the same systems.
        """
        key = model_info["key"]
        model_type = model_info["type"]

        if model_type == "extractive":
            return self.extractive.summarize(text, ratio=ratio, lang=lang)
        if model_type == "hf":
            return self._run_hf_model(self._resolve_hf_model_key(key, lang), text, ratio, lang)
        if model_type == "colab":
            from backend.summarizers.paid_models import ColabGPUClient

            # Translate the compression ratio into a token budget for the GPU
            # server, so remote models honour the same length target as local
            # ones instead of always generating the server's default length.
            words = len(text.split())
            target_words = max(50, int(words * ratio))
            max_length = min(512, max(80, int(target_words * 1.5)))
            min_length = max(30, int(max_length * 0.4))
            return ColabGPUClient().summarize(
                text, model=key, max_length=max_length, min_length=min_length
            )
        if model_type == "paid":
            return self.paid.summarize(text, model_name=key, lang=lang, ratio=ratio)
        raise ValueError(f"Unknown model type: {model_type}")

    def _resolve_hf_model_key(self, key: str, lang: str) -> str:
        """Resolve a UI model key to a MULTILINGUAL_MODELS key."""
        mapping = {
            "distilbart": "en",
            "pegasus": "en-pegasus",
            "bart-large": "en-bart-large",
            "mt5-xlsum": "multilingual",
            "mt5-turkish-finetuned": "tr-finetuned",
        }
        return mapping.get(key, "en" if lang == "en" else "multilingual")

    def _run_hf_model(self, model_key: str, text: str, ratio: float, lang: str) -> dict:
        """Run a specific HuggingFace model with hierarchical chunking for long texts."""
        config = MULTILINGUAL_MODELS[model_key]
        pipe, _ = self.abstractive._get_pipeline(lang, model_key)
        model_id = config["model_id"]
        is_mt5 = "mT5" in model_id or "mt5" in model_id

        preprocessed = self.preprocessor.preprocess(text, lang=lang)
        word_count = preprocessed["word_count"]

        max_length = max(100, int(word_count * ratio))
        min_length = max(60, int(max_length * 0.5))
        output_cap = 400 if not is_mt5 else 200
        max_length = min(max_length, output_cap)
        min_length = min(min_length, max_length - 20)
        min_length = max(min_length, 50)

        # Determine chunk limit based on model
        chunk_word_limit = 250 if is_mt5 else 750

        if word_count > chunk_word_limit:
            # Hierarchical chunking: chunk → summarize each → merge → summarize merged
            chunks = self.preprocessor.chunk_text(
                text, lang=lang,
                max_chunk_size=chunk_word_limit, overlap=30
            )

            # Summarize each chunk
            chunk_max = max(60, max_length // max(1, len(chunks) // 2))
            chunk_min = max(20, chunk_max // 3)
            chunk_summaries = []

            for chunk in chunks:
                if is_mt5:
                    import re
                    chunk = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', chunk.strip()))
                result = pipe(
                    chunk,
                    max_length=chunk_max,
                    min_length=chunk_min,
                    do_sample=False,
                    truncation=True,
                )
                chunk_summaries.append(result[0]["summary_text"])

            # Merge chunk summaries
            merged = " ".join(chunk_summaries)

            # Final summarization pass on merged text
            if is_mt5:
                import re
                merged = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', merged.strip()))

            merged_words = len(merged.split())
            if merged_words > chunk_word_limit:
                # Still too long, do recursive chunking
                chunks2 = self.preprocessor.chunk_text(
                    merged, lang=lang,
                    max_chunk_size=chunk_word_limit, overlap=20
                )
                chunk_summaries2 = []
                for c in chunks2:
                    r2 = pipe(c, max_length=chunk_max, min_length=chunk_min, do_sample=False, truncation=True)
                    chunk_summaries2.append(r2[0]["summary_text"])
                merged = " ".join(chunk_summaries2)

            final_result = pipe(
                merged,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            )
            summary = final_result[0]["summary_text"]
        else:
            input_text = preprocessed["cleaned_text"]
            if is_mt5:
                import re
                input_text = re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', input_text.strip()))

            result = pipe(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            )
            summary = result[0]["summary_text"]

        reliability = "high" if not is_mt5 else ("low" if word_count > chunk_word_limit else "medium")

        return {
            "summary": summary,
            "method": f"Transformer ({model_id.split('/')[-1]})",
            "model_id": model_id,
            "compression_ratio": len(summary.split()) / word_count if word_count > 0 else 0,
            "language": lang,
            "reliability": reliability,
            "warning": None if reliability == "high" else "May contain inaccuracies for long/non-English text.",
        }
