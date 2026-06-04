"""
Main Summarization Service (Multilingual).
Orchestrates preprocessing, summarization, and evaluation.
"""

from typing import Optional, Dict
from backend.preprocessing import TextPreprocessor
from backend.summarizers.extractive import ExtractiveSummarizer
from backend.summarizers.abstractive import AbstractiveSummarizer
from backend.evaluation.metrics import SummaryEvaluator


class SummarizationService:
    """
    Main service that coordinates the full summarization pipeline:
    1. Preprocess input text (with language detection)
    2. Generate summaries using multiple methods
    3. Evaluate and compare results
    """

    def __init__(self, model_name: str = "auto", use_bertscore: bool = False):
        self.preprocessor = TextPreprocessor()
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer(model_name=model_name)
        self.evaluator = SummaryEvaluator(use_bertscore=use_bertscore)

    def summarize_extractive(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: float = 0.3,
    ) -> Dict:
        """Generate extractive summary with evaluation."""
        result = self.extractive.summarize(text, num_sentences=num_sentences, ratio=ratio)
        lang = result.get("language", "en")
        metrics = self.evaluator.evaluate(text, result["summary"], lang=lang)
        return {**result, "metrics": metrics}

    def summarize_abstractive(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        ratio: float = 0.3,
    ) -> Dict:
        """Generate abstractive summary with evaluation."""
        result = self.abstractive.summarize(
            text, max_length=max_length, min_length=min_length, ratio=ratio
        )
        lang = result.get("language", "en")
        metrics = self.evaluator.evaluate(text, result["summary"], lang=lang)
        return {**result, "metrics": metrics}

    def summarize_and_compare(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: float = 0.3,
        reference_summary: Optional[str] = None,
    ) -> Dict:
        """
        Generate summaries using both methods and compare them.
        This is the main endpoint for the thesis comparison.
        """
        # Preprocess (detects language)
        preprocessed = self.preprocessor.preprocess(text)
        lang = preprocessed["language"]

        # Generate summaries with detected language
        extractive_result = self.extractive.summarize(
            text, num_sentences=num_sentences, ratio=ratio, lang=lang
        )
        abstractive_result = self.abstractive.summarize(text, ratio=ratio, lang=lang)

        # Evaluate and compare
        summaries = {
            "extractive_textrank": extractive_result["summary"],
            "abstractive_transformer": abstractive_result["summary"],
        }

        comparison = self.evaluator.compare_summaries(
            text, summaries, reference_summary=reference_summary, lang=lang
        )

        return {
            "input_stats": {
                "word_count": preprocessed["word_count"],
                "sentence_count": preprocessed["num_sentences"],
                "chunk_count": preprocessed["num_chunks"],
                "language": lang,
            },
            "extractive": {
                **extractive_result,
                "metrics": comparison["individual_results"]["extractive_textrank"],
            },
            "abstractive": {
                **abstractive_result,
                "metrics": comparison["individual_results"]["abstractive_transformer"],
            },
            "comparison": {
                "rankings": comparison["rankings"],
            },
        }
