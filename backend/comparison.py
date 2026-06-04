"""
Multi-Model Comparison Service.
Runs the same text through ALL available models and compares results.
"""

from typing import Optional, Dict, List
from backend.preprocessing import TextPreprocessor
from backend.summarizers.extractive import ExtractiveSummarizer
from backend.summarizers.abstractive import AbstractiveSummarizer, MULTILINGUAL_MODELS, EN_MODELS
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

    def get_available_models(self, lang: str = "en") -> List[dict]:
        """Get all models available for the given language."""
        models = [
            {"key": "extractive", "name": "TextRank (Extractive)", "type": "extractive"},
        ]

        # HuggingFace models — only include heavy models if Colab GPU is available
        from backend.summarizers.paid_models import ColabGPUClient
        colab = ColabGPUClient()
        colab_models = colab.get_available_models() if colab.is_available() else []

        if lang == "en":
            models.append({"key": "distilbart", "name": "DistilBART-CNN", "type": "hf"})
            # Only add heavy GPU models if Colab is running
            if "led-large-book" in colab_models:
                models.append({"key": "led-large-book", "name": "LED-Large-BookSum (GPU)", "type": "colab"})
            if "flan-t5-large" in colab_models:
                models.append({"key": "flan-t5-large", "name": "Flan-T5-Large (GPU)", "type": "colab"})
        else:
            models.append({"key": "mt5-xlsum", "name": "mT5-XLSum (Multilingual)", "type": "hf"})
            if "turkish-mbart" in colab_models and lang == "tr":
                models.append({"key": "turkish-mbart", "name": "mBART Turkish (GPU)", "type": "colab"})
            models.append({"key": "mt5-xlsum", "name": "mT5-XLSum (Multilingual)", "type": "hf"})

        # Paid models
        for model_key in self.paid.get_available_models():
            config = self.paid.SUPPORTED_MODELS[model_key]
            models.append({"key": model_key, "name": config["display_name"], "type": "paid"})

        return models

    def compare_all(
        self,
        text: str,
        reference_summary: Optional[str] = None,
        ratio: float = 0.3,
        models: Optional[List[str]] = None,
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

        available = self.get_available_models(lang)
        if models:
            available = [m for m in available if m["key"] in models]

        results = {}
        errors = {}

        for model_info in available:
            key = model_info["key"]
            try:
                if model_info["type"] == "extractive":
                    result = self.extractive.summarize(text, ratio=ratio, lang=lang)
                elif model_info["type"] == "hf":
                    if key == "distilbart":
                        result = self._run_hf_model("en", text, ratio, lang)
                    elif key == "mt5-xlsum":
                        result = self._run_hf_model("multilingual", text, ratio, lang)
                    else:
                        result = self.abstractive.summarize(text, ratio=ratio, lang=lang)
                elif model_info["type"] == "colab":
                    from backend.summarizers.paid_models import ColabGPUClient
                    colab = ColabGPUClient()
                    result = colab.summarize(text, model=key)
                elif model_info["type"] == "paid":
                    result = self.paid.summarize(text, model_name=key, lang=lang, ratio=ratio)
                else:
                    continue

                results[key] = {
                    "name": model_info["name"],
                    "summary": result["summary"],
                    "compression_ratio": result.get("compression_ratio", 0),
                    "method": result.get("method", model_info["name"]),
                    "warning": result.get("warning"),
                    "reliability": result.get("reliability", "unknown"),
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

    def _resolve_hf_model_key(self, key: str, lang: str) -> str:
        """Resolve a UI model key to a MULTILINGUAL_MODELS key."""
        mapping = {
            "distilbart": "en",
            "pegasus": "en-pegasus",
            "bart-large": "en-bart-large",
            "mt5-xlsum": "multilingual",
        }
        return mapping.get(key, "en")

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
