"""
Text Preprocessing Module (Multilingual)
Handles cleaning, tokenization, and preparation of input text for summarization.
Supports: English, Polish, Turkish, Azerbaijani, and other languages.
"""

import re
import nltk
from typing import List, Optional
from langdetect import detect, LangDetectException

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# NLTK stopwords language mapping
NLTK_LANG_MAP = {
    "en": "english",
    "pl": "polish",  # Polish is supported by NLTK stopwords
    "tr": "turkish",  # Turkish is supported by NLTK stopwords
    "az": "azerbaijani",  # Azerbaijani is supported by NLTK stopwords
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "nl": "dutch",
    "ar": "arabic",
}

# NLTK punkt tokenizer language mapping
PUNKT_LANG_MAP = {
    "en": "english",
    "pl": "polish",
    "tr": "turkish",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ru": "russian",
    "nl": "dutch",
}


class TextPreprocessor:
    """Multilingual text preprocessor for summarization pipelines."""

    def __init__(self):
        pass

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text. Returns ISO 639-1 code."""
        try:
            lang = detect(text)
            return lang
        except LangDetectException:
            return "en"  # Default to English

    def get_stopwords(self, lang: str) -> set:
        """Get stopwords for the detected language."""
        nltk_lang = NLTK_LANG_MAP.get(lang)
        if nltk_lang:
            try:
                words = set(nltk.corpus.stopwords.words(nltk_lang))
                return words
            except OSError:
                pass

        # Fallback: return empty set (no stopword filtering)
        return set()

    def clean_text(self, text: str) -> str:
        """Remove noise from text while preserving structure."""
        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)
        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep sentence punctuation and unicode letters
        text = re.sub(r"[^\w\s.,!?;:\-\'\"\u00C0-\u024F\u0400-\u04FF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]", "", text)
        return text.strip()

    def split_into_sentences(self, text: str, lang: str = "en") -> List[str]:
        """Split text into sentences using language-appropriate tokenizer."""
        punkt_lang = PUNKT_LANG_MAP.get(lang, "english")
        try:
            sentences = nltk.sent_tokenize(text, language=punkt_lang)
        except LookupError:
            # Fallback to basic regex splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out very short sentences (likely noise)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def chunk_text(self, text: str, lang: str = "en", max_chunk_size: int = 1024, overlap: int = 100) -> List[str]:
        """
        Split long text into overlapping chunks for processing.
        This is critical for handling long documents that exceed model token limits.
        """
        sentences = self.split_into_sentences(text, lang)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap sentences for context continuity
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s.split()) > overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s.split())
                current_chunk = overlap_sentences
                current_length = overlap_length
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

    def preprocess(self, text: str, lang: Optional[str] = None) -> dict:
        """
        Full preprocessing pipeline.
        Returns cleaned text, sentences, chunks, and detected language.
        """
        cleaned = self.clean_text(text)

        # Detect language if not provided
        if lang is None:
            lang = self.detect_language(cleaned)

        sentences = self.split_into_sentences(cleaned, lang)
        chunks = self.chunk_text(cleaned, lang)

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "sentences": sentences,
            "chunks": chunks,
            "num_sentences": len(sentences),
            "num_chunks": len(chunks),
            "word_count": len(cleaned.split()),
            "language": lang,
        }
