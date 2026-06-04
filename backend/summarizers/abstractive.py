"""
Abstractive Summarization using Transformer Models (Multilingual).
Supports English (BART) and multilingual (mBART/mT5) models.
"""

import re
from typing import Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from backend.preprocessing import TextPreprocessor

# Whitespace normalization required by mT5 XLSum
WHITESPACE_HANDLER = lambda k: re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', k.strip()))


# Models per language group
MULTILINGUAL_MODELS = {
    # English-optimized (default, fast)
    "en": {
        "model_id": "sshleifer/distilbart-cnn-12-6",
        "prefix": "",
        "max_input": 1024,
    },
    # English-optimized (higher quality, slower)
    "en-pegasus": {
        "model_id": "google/pegasus-cnn_dailymail",
        "prefix": "",
        "max_input": 1024,
    },
    # English BART large (highest quality open-source for English)
    "en-bart-large": {
        "model_id": "facebook/bart-large-cnn",
        "prefix": "",
        "max_input": 1024,
    },
    # Multilingual (covers Polish, Turkish, Azerbaijani, and 50+ languages)
    "multilingual": {
        "model_id": "csebuetnlp/mT5_multilingual_XLSum",
        "prefix": "",
        "max_input": 512,
    },
}

# Languages that should use the multilingual model
MULTILINGUAL_LANGS = {"pl", "tr", "az", "de", "fr", "es", "it", "pt", "ru", "ar", "nl", "uk", "cs", "hu", "ro"}

# All available HuggingFace model keys for English
EN_MODELS = ["en", "en-pegasus", "en-bart-large"]


class AbstractiveSummarizer:
    """
    Transformer-based abstractive summarizer with multilingual support.
    Uses distilBART for English and mT5 (XLSum) for other languages.
    """

    def __init__(self, model_name: str = "auto"):
        """
        Args:
            model_name: 'auto' for language-based selection, or a specific model ID
        """
        self.preprocessor = TextPreprocessor()
        self.model_name = model_name
        self.method_name = "Transformer (Abstractive)"

        # Loaded models are kept in memory to avoid reloading on each request
        self._pipelines = {}

    def _get_model_config(self, lang: str, model_key: Optional[str] = None) -> dict:
        """Get the appropriate model config for the language or specific model key."""
        # If a specific model key is requested, use it directly
        if model_key and model_key in MULTILINGUAL_MODELS:
            return MULTILINGUAL_MODELS[model_key]
        # Auto-select based on language
        if lang == "en":
            return MULTILINGUAL_MODELS["en"]
        if lang in MULTILINGUAL_LANGS:
            return MULTILINGUAL_MODELS["multilingual"]
        # Default to multilingual for unknown languages
        return MULTILINGUAL_MODELS["multilingual"]

    def _get_pipeline(self, lang: str, model_key: Optional[str] = None):
        """Get or create the summarization pipeline for the given language."""
        config = self._get_model_config(lang, model_key)
        model_id = config["model_id"]

        if model_id not in self._pipelines:
            tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=config["max_input"])
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            self._pipelines[model_id] = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=-1,  # CPU; set to 0 for GPU
            )

        return self._pipelines[model_id], config

    @property
    def _pipeline(self):
        """Compatibility property — check if any model is loaded."""
        return self._pipelines.get(MULTILINGUAL_MODELS["en"]["model_id"])

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        ratio: float = 0.3,
        lang: Optional[str] = None,
    ) -> dict:
        """
        Generate an abstractive summary.

        Args:
            text: Input text to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            ratio: Target compression ratio
            lang: Language code (auto-detected if None)

        Returns:
            Dictionary with summary and metadata
        """
        preprocessed = self.preprocessor.preprocess(text, lang=lang)
        lang = preprocessed["language"]
        word_count = preprocessed["word_count"]

        pipe, config = self._get_pipeline(lang)
        model_id = config["model_id"]
        max_input = config["max_input"]
        is_mt5 = "mT5" in model_id or "mt5" in model_id

        # Set default lengths based on input — be more generous
        if max_length is None:
            max_length = max(100, int(word_count * ratio))
        if min_length is None:
            min_length = max(60, int(max_length * 0.5))

        # Cap to model's actual output capacity
        output_cap = 400 if not is_mt5 else 200
        max_length = min(max_length, output_cap)
        min_length = min(min_length, max_length - 20)
        min_length = max(min_length, 50)

        # Determine if text needs chunking
        chunk_word_limit = 250 if is_mt5 else 750
        needs_chunking = word_count > chunk_word_limit

        if needs_chunking:
            # HIERARCHICAL CHUNKING for long documents:
            # 1. Split text into chunks
            # 2. Summarize each chunk
            # 3. Merge chunk summaries
            # 4. Summarize merged text for final coherent summary
            chunks = self.preprocessor.chunk_text(
                text, lang=lang,
                max_chunk_size=chunk_word_limit, overlap=30
            )
            summary = self._summarize_long_document(
                chunks, max_length, min_length, ratio, lang
            )
        else:
            input_text = preprocessed["cleaned_text"]
            if is_mt5:
                input_text = WHITESPACE_HANDLER(input_text)
            result = pipe(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            )
            summary = result[0]["summary_text"]

        self.method_name = f"Transformer ({model_id.split('/')[-1]})"

        # Reliability warning for long non-English text
        reliability = "high"
        warning = None
        if is_mt5 and needs_chunking:
            reliability = "low"
            warning = (
                "⚠️ This summary may contain inaccuracies. The multilingual model (mT5) "
                "is prone to hallucination on long documents. Verify facts against the source text."
            )
        elif is_mt5:
            reliability = "medium"
            warning = "This summary was generated by a multilingual model. Minor inaccuracies are possible."

        return {
            "summary": summary,
            "method": self.method_name,
            "model_id": model_id,
            "max_length": max_length,
            "min_length": min_length,
            "compression_ratio": len(summary.split()) / word_count if word_count > 0 else 0,
            "language": lang,
            "reliability": reliability,
            "warning": warning,
        }

    def _summarize_long_document(
        self, chunks: list, max_length: int, min_length: int, ratio: float, lang: str
    ) -> str:
        """
        Summarize long documents using hierarchical chunking:
        1. Summarize each chunk individually
        2. Merge chunk summaries
        3. Summarize the merged text for a final coherent summary
        """
        pipe, config = self._get_pipeline(lang)
        model_id = config["model_id"]
        is_mt5 = "mT5" in model_id or "mt5" in model_id
        chunk_word_limit = 250 if is_mt5 else 500

        chunk_summaries = []
        # Each chunk gets a proportional share of the output length
        chunk_max = max(30, min(80, int(max_length / max(1, len(chunks) // 3))))
        chunk_min = max(10, int(chunk_max * 0.3))

        for chunk in chunks:
            input_chunk = WHITESPACE_HANDLER(chunk) if is_mt5 else chunk
            result = pipe(
                input_chunk,
                max_length=chunk_max,
                min_length=chunk_min,
                do_sample=False,
                truncation=True,
            )
            chunk_summaries.append(result[0]["summary_text"])

        # Merge chunk summaries
        merged = " ".join(chunk_summaries)

        # If merged is still too long for one pass, recurse
        merged_words = len(merged.split())
        if merged_words > chunk_word_limit:
            sub_chunks = self.preprocessor.chunk_text(
                merged, lang=lang,
                max_chunk_size=chunk_word_limit, overlap=20
            )
            return self._summarize_long_document(
                sub_chunks, max_length, min_length, ratio, lang
            )

        # Final summarization pass on merged chunk summaries
        input_merged = WHITESPACE_HANDLER(merged) if is_mt5 else merged
        result = pipe(
            input_merged,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"]
