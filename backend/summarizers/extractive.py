"""
Extractive Summarization using TextRank Algorithm (Multilingual).
Selects the most important sentences from the original text based on graph-based ranking.
"""

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

from backend.preprocessing import TextPreprocessor


class ExtractiveSummarizer:
    """
    TextRank-based extractive summarizer.
    Builds a graph of sentences weighted by similarity,
    then uses PageRank to identify the most important sentences.
    Supports multiple languages via language-aware stopwords.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.method_name = "TextRank (Extractive)"

    def _build_similarity_matrix(self, sentences: List[str], lang: str = "en") -> np.ndarray:
        """Build a cosine similarity matrix using TF-IDF vectors."""
        stopwords = self.preprocessor.get_stopwords(lang)
        # Use custom stopwords list or None if empty
        sw = list(stopwords) if stopwords else None
        vectorizer = TfidfVectorizer(stop_words=sw)
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # If vocabulary is empty after stopword removal, try without stopwords
            vectorizer = TfidfVectorizer(stop_words=None)
            tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        return similarity_matrix

    def _textrank(self, sentences: List[str], top_n: int, lang: str = "en") -> List[int]:
        """Apply TextRank algorithm with position and narrative boosting."""
        if len(sentences) <= top_n:
            return list(range(len(sentences)))

        similarity_matrix = self._build_similarity_matrix(sentences, lang)

        # Build graph and apply PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        try:
            scores = nx.pagerank(graph, max_iter=100)
        except nx.PowerIterationFailedConvergence:
            scores = {i: 1.0 / len(sentences) for i in range(len(sentences))}

        # Boost scores based on position and content
        boosted_scores = {}
        for i, sentence in enumerate(sentences):
            base_score = scores[i]

            # Position boost: first and last sentences of the document are important
            position_ratio = i / max(len(sentences) - 1, 1)
            if position_ratio < 0.1:  # First 10%
                base_score *= 1.3
            elif position_ratio > 0.9:  # Last 10%
                base_score *= 1.2

            # Length boost: medium-length sentences are usually more informative
            word_count = len(sentence.split())
            if 12 <= word_count <= 35:
                base_score *= 1.1

            # Narrative/causal boost: sentences with plot-driving words
            causal_words = ["because", "but", "however", "instead", "finally",
                           "after", "before", "when", "then", "so",
                           "çünkü", "ama", "ancak", "sonra", "önce",
                           "daha sonra", "bunun üzerine", "oysa"]
            lower_sent = sentence.lower()
            if any(w in lower_sent for w in causal_words):
                base_score *= 1.2

            # Named entity density boost (sentences with proper nouns carry more info)
            capitalized_words = sum(1 for w in sentence.split() if w[0:1].isupper() and len(w) > 1)
            if capitalized_words >= 2:
                base_score *= 1.05

            boosted_scores[i] = base_score

        # Get top-n sentence indices, sorted by original position
        ranked_indices = sorted(boosted_scores, key=boosted_scores.get, reverse=True)[:top_n]
        return sorted(ranked_indices)

    def _is_header(self, sentence: str) -> bool:
        """Detect if a sentence is actually a section header (not useful for summary)."""
        s = sentence.strip()
        # All uppercase or mostly uppercase
        if s.isupper() and len(s) < 100:
            return True
        # Very short with no verb-like punctuation
        if len(s.split()) < 5 and not any(p in s for p in ['.', ',', ';']):
            return True
        # Starts with a known header pattern
        if s.startswith(("BİR ", "ATATÜRK'ÜN ", "ATATÜRKÜN ")) and len(s.split()) < 10:
            return True
        return False

    def _is_self_contained(self, sentence: str) -> bool:
        """
        Check if a sentence is self-contained (understandable without context).
        Filters out sentences that start with pronouns or connectors without a subject.
        """
        s = sentence.strip()
        # Too short to be meaningful
        if len(s.split()) < 6:
            return False
        # Starts with a pronoun/connector that needs prior context
        weak_starts = [
            "O ", "Onu ", "Ona ", "Onun ", "Bu ", "Bunu ", "Buna ",
            "Bunun ", "Böyle ", "Böylelikle ", "Ancak ", "Oysa ",
            "He ", "She ", "It ", "They ", "This ", "That ", "These ",
            "However ", "But ", "Also ", "Then ",
        ]
        for start in weak_starts:
            if s.startswith(start) and len(s.split()) < 12:
                return False
        return True

    def _filter_sentences(self, sentences: List[str]) -> List[str]:
        """Filter out headers and non-self-contained sentences."""
        filtered = []
        for s in sentences:
            if self._is_header(s):
                continue
            if not self._is_self_contained(s):
                continue
            filtered.append(s)
        # If filtering removed too many, fall back to original
        if len(filtered) < max(3, len(sentences) // 4):
            return sentences
        return filtered

    def summarize(
        self,
        text: str,
        num_sentences: Optional[int] = None,
        ratio: float = 0.3,
        lang: Optional[str] = None,
    ) -> dict:
        """
        Generate an extractive summary.

        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract (overrides ratio)
            ratio: Proportion of sentences to keep (default 30%)
            lang: Language code (auto-detected if None)

        Returns:
            Dictionary with summary and metadata
        """
        preprocessed = self.preprocessor.preprocess(text, lang=lang)
        lang = preprocessed["language"]
        sentences = preprocessed["sentences"]

        if not sentences:
            return {
                "summary": text,
                "method": self.method_name,
                "num_sentences_selected": 0,
                "num_sentences_total": 0,
                "language": lang,
            }

        # Filter out headers and context-dependent sentences
        filtered_sentences = self._filter_sentences(sentences)

        # Determine number of sentences to extract
        if num_sentences is None:
            num_sentences = max(1, int(len(filtered_sentences) * ratio))
        num_sentences = min(num_sentences, len(filtered_sentences))

        # For long documents, process globally (not per-chunk)
        if len(filtered_sentences) > 20:
            summary = self._summarize_long_document_global(filtered_sentences, num_sentences, lang)
        else:
            selected_indices = self._textrank(filtered_sentences, num_sentences, lang)
            summary = " ".join([filtered_sentences[i] for i in selected_indices])

        return {
            "summary": summary,
            "method": self.method_name,
            "num_sentences_selected": num_sentences,
            "num_sentences_total": len(sentences),
            "compression_ratio": len(summary.split()) / preprocessed["word_count"] if preprocessed["word_count"] > 0 else 0,
            "language": lang,
            # Sentences are copied verbatim from the source, so the output
            # cannot contain fabricated content at sentence level.
            "reliability": "high",
            "warning": None,
        }

    def _summarize_long_document_global(self, sentences: List[str], num_sentences: int, lang: str) -> str:
        """
        Summarize a long document using global TextRank on all filtered sentences.
        No chunking — applies TextRank on the full sentence set for better coherence.
        """
        selected_indices = self._textrank(sentences, num_sentences, lang)

        # Deduplicate by content similarity
        selected_sentences = []
        seen = set()
        for i in selected_indices:
            normalized = sentences[i].lower()[:80]
            if normalized not in seen:
                seen.add(normalized)
                selected_sentences.append(sentences[i])

        return " ".join(selected_sentences)

    def _summarize_long_document(self, chunks: List[str], ratio: float, lang: str) -> str:
        """Legacy chunk-based method (kept for compatibility)."""
        all_sentences = []
        for chunk in chunks:
            sentences = self.preprocessor.split_into_sentences(chunk, lang)
            all_sentences.extend(sentences)

        # Deduplicate
        unique_sentences = []
        seen = set()
        for s in all_sentences:
            normalized = s.lower()[:80]
            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(s)

        # Filter
        filtered = self._filter_sentences(unique_sentences)
        num_sentences = max(3, int(len(filtered) * ratio))
        return self._summarize_long_document_global(filtered, num_sentences, lang)
