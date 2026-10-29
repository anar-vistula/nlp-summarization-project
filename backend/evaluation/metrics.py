"""
Summary Evaluation Metrics (Multilingual).

The metric set is split into two families, which are never mixed:

  Reference-based (need a human-written reference summary)
    - ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum  (Lin, 2004)
    - METEOR                                     (Banerjee & Lavie, 2005)
    - BERTScore, optional                        (Zhang et al., 2020)

  Intrinsic / reference-free (source text is the only ground truth)
    - Compression ratio
    - Information density (retention of top source TF-IDF terms)
    - Topic coverage (retention of top source key phrases)
    - Source coverage (fraction of source sentences represented)
    - Lexical cohesion (adjacent-sentence similarity; proxy for coherence)
    - Redundancy + distinct-n repetition
    - Abstractiveness: novel n-gram ratio           (See et al., 2017)
    - Extractive fragment coverage & density        (Grusky et al., 2018)
    - Factual-consistency proxy: unsupported numbers and entities
    - Readability: Flesch Reading Ease (en) and LIX

Two deliberate methodological choices:

1. ROUGE is reported ONLY against a human reference. Earlier versions of this
   module fell back to scoring the summary against the full source text and
   labelled the result "ROUGE", which is not what ROUGE measures: recall is
   driven to near zero by the length of the source, and precision merely
   counts copying. That quantity is still useful, so it is computed and
   reported separately under `source_overlap` with its own name.

2. All tokenization is Unicode-aware (see text_utils), because the default
   rouge_score tokenizer deletes non-ASCII characters and would score every
   Turkish summary against a mangled version of itself.
"""

import numpy as np
from typing import Dict, Optional, List
from rouge_score import rouge_scorer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.preprocessing.text_preprocessor import TextPreprocessor
from backend.evaluation.text_utils import (
    UnicodeRougeTokenizer,
    count_words,
    match_token_set,
    count_syllables_en,
    extract_capitalized_phrases,
    extract_numbers,
    ngram_set,
    ngrams,
    normalize_for_match,
    phrase_in_text,
    safe_div,
    sentence_split,
    term_in_tokens,
    tokenize,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# --- Tunable thresholds (documented so results are reproducible) ---------
# A source sentence counts as "covered" when its cosine similarity to the most
# similar summary sentence exceeds this value.
COVERAGE_SIM_THRESHOLD = 0.30
# Two summary sentences count as a redundant pair above this cosine similarity.
REDUNDANCY_SIM_THRESHOLD = 0.50
# Number of top TF-IDF terms treated as the "key information" of the source.
N_IMPORTANT_TERMS = 50
# Number of top key phrases (uni- + bi-grams) treated as the source's topics.
N_TOPICS = 20

# --- Composite score weights --------------------------------------------
# Any weighting of heterogeneous metrics is a modelling choice, not a fact.
# These weights are stated explicitly, reported with every result, and the
# individual components are always available so a reader can re-weight them.
INTRINSIC_WEIGHTS = {
    "information_density": 0.20,
    "factual_consistency": 0.25,
    "source_coverage": 0.15,
    "topic_coverage": 0.15,
    "non_redundancy": 0.15,
    "lexical_cohesion": 0.10,
}
REFERENCE_WEIGHTS = {
    "rouge_1_f1": 0.20,
    "rouge_2_f1": 0.20,
    "rouge_l_f1": 0.25,
    "meteor": 0.15,
    "bert_score_f1": 0.20,
}


class SummaryEvaluator:
    """Comprehensive multilingual summary evaluation."""

    def __init__(self, use_bertscore: bool = False):
        """
        Args:
            use_bertscore: Whether to compute BERTScore (slower, needs a model
                download on first use).
        """
        self.use_bertscore = use_bertscore
        self.preprocessor = TextPreprocessor()
        self._rouge_scorers: Dict[str, rouge_scorer.RougeScorer] = {}

    # ------------------------------------------------------------------ #
    # Entry points
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        source_text: str,
        summary: str,
        reference_summary: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate one summary.

        Args:
            source_text: Original input text.
            summary: Generated summary to evaluate.
            reference_summary: Optional human-written reference summary.
                Without it, the reference-based metrics are omitted rather
                than approximated.
            lang: Language code (auto-detected when None).

        Returns:
            Dictionary of all computed metrics.
        """
        if lang is None:
            lang = self.preprocessor.detect_language(source_text)

        stopwords = self.preprocessor.get_stopwords(lang)
        sw_list = list(stopwords) if stopwords else None

        source_tokens = tokenize(source_text, lang=lang)
        summary_tokens = tokenize(summary, lang=lang)

        metrics: Dict = {"language": lang}
        has_reference = bool(reference_summary and reference_summary.strip())
        metrics["reference_available"] = has_reference

        # --- Intrinsic metrics (always computable) ---
        metrics["compression_ratio"] = self._compression_ratio(source_text, summary)
        metrics["information_density"] = self._information_density(
            source_text, summary_tokens, sw_list, lang
        )
        metrics["redundancy_score"] = self._redundancy_score(summary, sw_list, lang)
        metrics["coherence_score"] = self._coherence_score(summary, sw_list, lang)
        metrics["coverage_score"] = self._coverage_score(source_text, summary, sw_list, lang)
        metrics["length_stats"] = self._length_statistics(source_text, summary, lang)
        metrics["topic_coverage"] = self._topic_coverage(
            source_text, summary_tokens, sw_list, lang
        )
        metrics["abstractiveness"] = self._abstractiveness(source_tokens, summary_tokens)
        metrics["extractive_fragments"] = self._extractive_fragments(
            source_tokens, summary_tokens
        )
        metrics["factual_consistency"] = self._factual_consistency(
            source_text, summary, lang
        )
        metrics["readability"] = self._readability(summary, lang)
        # Overlap with the source. Useful, but deliberately NOT called ROUGE.
        metrics["source_overlap"] = self._source_overlap(source_text, summary, lang)

        # --- Reference-based metrics (only with a real reference) ---
        if has_reference:
            metrics["rouge_scores"] = self._compute_rouge(reference_summary, summary, lang)
            metrics["meteor_score"] = self._compute_meteor(reference_summary, summary, lang)
            if self.use_bertscore:
                metrics["bert_score"] = self._compute_bertscore(
                    reference_summary, summary, lang
                )

        # --- Composite scores ---
        metrics["overall_quality"] = self._compute_overall_quality(metrics)

        # --- Descriptions and human-readable explanations ---
        metrics["metric_metadata"] = self.metric_metadata()
        metrics["explanations"] = self._generate_explanations(metrics)

        return metrics

    def compare_summaries(
        self,
        source_text: str,
        summaries: Dict[str, str],
        reference_summary: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate several summaries of the same source and rank them.

        Args:
            source_text: Original text.
            summaries: Mapping of method name -> summary text.
            reference_summary: Optional reference summary.
            lang: Language code.
        """
        results = {}
        for method_name, summary in summaries.items():
            results[method_name] = self.evaluate(
                source_text, summary, reference_summary, lang=lang
            )
        return {
            "individual_results": results,
            "rankings": self._rank_summaries(results),
        }

    # ------------------------------------------------------------------ #
    # Reference-based metrics
    # ------------------------------------------------------------------ #

    def _get_rouge_scorer(self, lang: str) -> rouge_scorer.RougeScorer:
        """
        ROUGE scorer with a Unicode-aware tokenizer.

        Stemming is applied for English only — the Porter stemmer is an
        English algorithm and mangles Turkish surface forms.
        """
        if lang not in self._rouge_scorers:
            self._rouge_scorers[lang] = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL", "rougeLsum"],
                tokenizer=UnicodeRougeTokenizer(lang=lang, stem=(lang == "en")),
                split_summaries=True,
            )
        return self._rouge_scorers[lang]

    def _compute_rouge(self, reference: str, summary: str, lang: str = "en") -> Dict:
        """ROUGE-1/2/L/Lsum of the summary against a human reference."""
        scores = self._get_rouge_scorer(lang).score(reference, summary)
        out = {}
        for key in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
            score = scores[key]
            out[key] = {
                "precision": round(score.precision, 4),
                "recall": round(score.recall, 4),
                "f1": round(score.fmeasure, 4),
            }
        out["basis"] = "human reference summary"
        return out

    def _compute_meteor(self, reference: str, summary: str, lang: str = "en") -> Dict:
        """
        METEOR: unigram alignment with stem and WordNet-synonym matching.
        Synonym matching only exists for English; for other languages the
        score degrades to exact/stem matching, which is noted in the output.
        """
        try:
            from nltk.translate.meteor_score import meteor_score

            hypothesis = tokenize(summary, lang=lang)
            ref_tokens = tokenize(reference, lang=lang)
            if not hypothesis or not ref_tokens:
                return {"score": 0.0, "synonym_matching": lang == "en"}
            score = meteor_score([ref_tokens], hypothesis)
            return {
                "score": round(float(score), 4),
                "synonym_matching": lang == "en",
            }
        except Exception as exc:  # missing wordnet data, etc.
            return {"score": 0.0, "error": str(exc), "synonym_matching": False}

    def _compute_bertscore(self, reference: str, summary: str, lang: str = "en") -> Dict:
        """
        BERTScore against the reference summary, using a model appropriate to
        the detected language.
        """
        try:
            from bert_score import score as bert_score

            P, R, F1 = bert_score([summary], [reference], lang=lang, verbose=False)
            return {
                "precision": round(float(P[0]), 4),
                "recall": round(float(R[0]), 4),
                "f1": round(float(F1[0]), 4),
            }
        except Exception as exc:
            return {"error": str(exc), "precision": 0, "recall": 0, "f1": 0}

    # ------------------------------------------------------------------ #
    # Intrinsic metrics
    # ------------------------------------------------------------------ #

    def _source_overlap(self, source: str, summary: str, lang: str) -> Dict:
        """
        N-gram overlap between summary and source text.

        This is NOT ROUGE: the "reference" here is the whole document, so
        recall is bounded by the compression ratio and carries no quality
        information. Precision, however, is meaningful — it is the fraction of
        the summary's n-grams that were copied from the source, i.e. a direct
        measure of extractiveness.
        """
        scores = self._get_rouge_scorer(lang).score(source, summary)
        return {
            "unigram_copy_rate": round(scores["rouge1"].precision, 4),
            "bigram_copy_rate": round(scores["rouge2"].precision, 4),
            "note": (
                "Precision against the full source = copy rate. Recall is not "
                "reported because it is determined by document length, not quality."
            ),
        }

    def _compression_ratio(self, source: str, summary: str) -> Dict:
        """Compression ratio. Lower = more compression."""
        source_words = count_words(source)
        summary_words = count_words(summary)
        ratio = safe_div(summary_words, source_words)
        return {
            "ratio": round(ratio, 4),
            "source_words": source_words,
            "summary_words": summary_words,
            "reduction_percentage": round((1 - ratio) * 100, 2),
        }

    def _top_source_terms(
        self,
        source: str,
        stopwords: Optional[list],
        lang: str,
        max_features: int,
        ngram_range=(1, 1),
    ) -> List[str]:
        """
        Highest-weight terms of the source by summed TF-IDF.

        TF-IDF is fitted over the source's *sentences*, so the IDF component is
        real: a term that appears in every sentence is down-weighted. Fitting
        on the document as a single string (as an earlier version did) makes
        IDF constant and reduces the ranking to raw frequency.
        """
        sentences = sentence_split(source, lang)
        corpus = sentences if len(sentences) >= 2 else [source]
        try:
            vectorizer = TfidfVectorizer(
                stop_words=stopwords, ngram_range=ngram_range, max_features=2000
            )
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            try:
                vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=2000)
                matrix = vectorizer.fit_transform(corpus)
            except ValueError:
                return []
        weights = np.asarray(matrix.sum(axis=0)).ravel()
        features = vectorizer.get_feature_names_out()
        order = np.argsort(-weights)[:max_features]
        return [str(features[i]) for i in order]

    def _information_density(
        self,
        source: str,
        summary_tokens: List[str],
        stopwords: Optional[list],
        lang: str,
    ) -> Dict:
        """
        Fraction of the source's most important terms that survive in the
        summary. Matching is token-level with a prefix rule for inflected
        forms, not raw substring search.
        """
        terms = self._top_source_terms(source, stopwords, lang, N_IMPORTANT_TERMS)
        if not terms:
            return {
                "score": 0.0,
                "important_terms_total": 0,
                "important_terms_in_summary": 0,
                "missing_terms": [],
            }
        token_set = match_token_set(summary_tokens)
        present = [t for t in terms if term_in_tokens(t, token_set)]
        return {
            "score": round(safe_div(len(present), len(terms)), 4),
            "important_terms_total": len(terms),
            "important_terms_in_summary": len(present),
            "missing_terms": [t for t in terms if t not in present][:15],
        }

    def _topic_coverage(
        self,
        source: str,
        summary_tokens: List[str],
        stopwords: Optional[list],
        lang: str,
    ) -> Dict:
        """Presence of the source's top key phrases (uni- and bi-grams)."""
        topics = self._top_source_terms(
            source, stopwords, lang, N_TOPICS, ngram_range=(1, 2)
        )
        if not topics:
            return {
                "score": 0.0,
                "total_topics": 0,
                "covered_topics": [],
                "missing_topics": [],
            }
        token_set = match_token_set(summary_tokens)
        covered = [t for t in topics if term_in_tokens(t, token_set)]
        missing = [t for t in topics if t not in covered]
        return {
            "score": round(safe_div(len(covered), len(topics)), 4),
            "total_topics": len(topics),
            "covered_topics": covered,
            "missing_topics": missing,
        }

    def _redundancy_score(
        self, summary: str, stopwords: Optional[list], lang: str = "en"
    ) -> Dict:
        """
        Internal repetition of the summary. Lower is better.

        Two views: sentence-level (pairs above a similarity threshold) and
        token-level distinct-n, which catches the degenerate n-gram loops
        transformer decoders produce on chunked long documents.
        """
        sentences = sentence_split(summary, lang)
        tokens = tokenize(summary, lang=lang)
        distinct = {}
        for n in (1, 2, 3):
            grams = ngrams(tokens, n)
            distinct[f"distinct_{n}"] = round(
                safe_div(len(set(grams)), len(grams), default=1.0), 4
            )
        repeated_4gram_rate = 0.0
        grams4 = ngrams(tokens, 4)
        if grams4:
            repeated_4gram_rate = round(
                1.0 - safe_div(len(set(grams4)), len(grams4), default=1.0), 4
            )

        base = {
            "num_sentences": len(sentences),
            "repeated_4gram_rate": repeated_4gram_rate,
            **distinct,
        }
        if len(sentences) <= 1:
            return {"score": 0.0, "redundant_pairs": 0, "max_similarity": 0.0, **base}

        try:
            vectorizer = TfidfVectorizer(stop_words=stopwords)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sim_matrix = cosine_similarity(tfidf_matrix)
            np.fill_diagonal(sim_matrix, 0)
            redundant_pairs = int(np.sum(sim_matrix > REDUNDANCY_SIM_THRESHOLD) / 2)
            total_pairs = len(sentences) * (len(sentences) - 1) / 2
            return {
                "score": round(safe_div(redundant_pairs, total_pairs), 4),
                "redundant_pairs": redundant_pairs,
                "max_similarity": round(float(sim_matrix.max()), 4),
                **base,
            }
        except ValueError:
            return {"score": 0.0, "redundant_pairs": 0, "max_similarity": 0.0, **base}

    def _coherence_score(
        self, summary: str, stopwords: Optional[list], lang: str = "en"
    ) -> Dict:
        """
        Lexical cohesion between consecutive sentences: mean cosine similarity
        of adjacent sentence pairs, plus a stopword-free Jaccard overlap.

        This is a proxy for coherence, not coherence itself — it detects
        abrupt topic jumps (a known artefact of extractive selection and of
        chunk-merged abstractive output) but cannot judge logical flow.
        Values are low in absolute terms by nature and should only be compared
        between systems on the same document.
        """
        sentences = sentence_split(summary, lang)
        if len(sentences) <= 1:
            return {
                "score": 1.0,
                "num_transitions": 0,
                "avg_transition_similarity": 1.0,
                "jaccard": 1.0,
                "single_sentence": True,
            }

        stop_set = set(stopwords or [])
        jaccards = []
        for a, b in zip(sentences, sentences[1:]):
            ta = {t for t in tokenize(a, lang=lang) if t not in stop_set}
            tb = {t for t in tokenize(b, lang=lang) if t not in stop_set}
            union = ta | tb
            jaccards.append(safe_div(len(ta & tb), len(union)))

        try:
            vectorizer = TfidfVectorizer(stop_words=stopwords)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            transition_scores = [
                float(cosine_similarity(tfidf_matrix[i : i + 1], tfidf_matrix[i + 1 : i + 2])[0][0])
                for i in range(len(sentences) - 1)
            ]
        except ValueError:
            transition_scores = jaccards

        avg = float(np.mean(transition_scores)) if transition_scores else 0.0
        return {
            "score": round(avg, 4),
            "num_transitions": len(transition_scores),
            "avg_transition_similarity": round(avg, 4),
            "min_transition": round(float(min(transition_scores)), 4) if transition_scores else 0.0,
            "max_transition": round(float(max(transition_scores)), 4) if transition_scores else 0.0,
            "jaccard": round(float(np.mean(jaccards)), 4) if jaccards else 0.0,
            "single_sentence": False,
        }

    def _coverage_score(
        self, source: str, summary: str, stopwords: Optional[list], lang: str = "en"
    ) -> Dict:
        """
        Fraction of source sentences that have a similar sentence in the
        summary (cosine similarity above COVERAGE_SIM_THRESHOLD). Measures how
        much of the document is represented at all, rather than how well.
        """
        source_sentences = sentence_split(source, lang)
        summary_sentences = sentence_split(summary, lang)
        if not source_sentences or not summary_sentences:
            return {
                "score": 0.0,
                "covered_sentences": 0,
                "total_sentences": len(source_sentences),
                "avg_coverage": 0.0,
                "threshold": COVERAGE_SIM_THRESHOLD,
            }
        try:
            vectorizer = TfidfVectorizer(stop_words=stopwords)
            tfidf_matrix = vectorizer.fit_transform(source_sentences + summary_sentences)
            source_vectors = tfidf_matrix[: len(source_sentences)]
            summary_vectors = tfidf_matrix[len(source_sentences) :]
            max_similarities = cosine_similarity(source_vectors, summary_vectors).max(axis=1)
            covered = int(np.sum(max_similarities > COVERAGE_SIM_THRESHOLD))
            return {
                "score": round(safe_div(covered, len(source_sentences)), 4),
                "covered_sentences": covered,
                "total_sentences": len(source_sentences),
                "avg_coverage": round(float(max_similarities.mean()), 4),
                "threshold": COVERAGE_SIM_THRESHOLD,
            }
        except ValueError:
            return {
                "score": 0.0,
                "covered_sentences": 0,
                "total_sentences": len(source_sentences),
                "avg_coverage": 0.0,
                "threshold": COVERAGE_SIM_THRESHOLD,
            }

    def _abstractiveness(
        self, source_tokens: List[str], summary_tokens: List[str]
    ) -> Dict:
        """
        Novel n-gram ratio (See et al., 2017): the share of summary n-grams
        that do not occur in the source.

        This is the metric that separates the two paradigms this thesis
        compares: a purely extractive summary scores ~0 by construction, while
        genuine abstraction produces novel unigrams and many novel bigrams.
        High novelty is also where hallucination lives, so it is descriptive,
        not a quality score.
        """
        out: Dict = {}
        for n in (1, 2, 3, 4):
            summary_grams = ngrams(summary_tokens, n)
            source_grams = ngram_set(source_tokens, n)
            if not summary_grams:
                out[f"novel_{n}gram_ratio"] = 0.0
                continue
            novel = sum(1 for g in summary_grams if g not in source_grams)
            out[f"novel_{n}gram_ratio"] = round(safe_div(novel, len(summary_grams)), 4)
        out["score"] = out.get("novel_1gram_ratio", 0.0)
        out["interpretation"] = (
            "0 = fully copied from the source (extractive); higher = more "
            "rewriting (abstractive). Not a quality score."
        )
        return out

    def _extractive_fragments(
        self, source_tokens: List[str], summary_tokens: List[str]
    ) -> Dict:
        """
        Extractive fragment coverage and density (Grusky et al., 2018).

        Greedily matches the summary against the source as a sequence of
        maximal shared fragments:
          coverage = share of summary tokens that come from such fragments
          density  = mean squared fragment length (how *long* the copied runs
                     are, i.e. copy-paste vs. word-level reuse)
        """
        fragments = self._greedy_fragments(source_tokens, summary_tokens)
        n_summary = len(summary_tokens)
        total_len = sum(len(f) for f in fragments)
        squared = sum(len(f) ** 2 for f in fragments)
        return {
            "coverage": round(safe_div(total_len, n_summary), 4),
            "density": round(safe_div(squared, n_summary), 4),
            "longest_fragment": max((len(f) for f in fragments), default=0),
            "num_fragments": len(fragments),
            "compression": round(safe_div(len(source_tokens), n_summary), 2),
        }

    @staticmethod
    def _greedy_fragments(source_tokens: List[str], summary_tokens: List[str]) -> List[List[str]]:
        """Maximal shared token runs between summary and source (Grusky et al.)."""
        fragments: List[List[str]] = []
        source_index: Dict[str, List[int]] = {}
        for idx, tok in enumerate(source_tokens):
            source_index.setdefault(tok, []).append(idx)

        i = 0
        while i < len(summary_tokens):
            best: List[str] = []
            for j in source_index.get(summary_tokens[i], ()):
                i_, j_ = i, j
                while (
                    i_ < len(summary_tokens)
                    and j_ < len(source_tokens)
                    and summary_tokens[i_] == source_tokens[j_]
                ):
                    i_ += 1
                    j_ += 1
                if (i_ - i) > len(best):
                    best = summary_tokens[i:i_]
            if best:
                fragments.append(best)
                i += len(best)
            else:
                i += 1
        return fragments

    def _factual_consistency(self, source: str, summary: str, lang: str) -> Dict:
        """
        Proxy for factual consistency: every number and every proper-noun
        phrase in the summary should be traceable to the source.

        A summary that invents dates, quantities or names is unfaithful
        regardless of how well it scores on overlap metrics — the dominant
        failure mode observed for chunked long documents. This is a
        precision-style check (it cannot detect distorted relations between
        facts), so it is reported as a proxy with the offending items listed
        for manual inspection.
        """
        source_norm = normalize_for_match(source)
        source_numbers = set(extract_numbers(source))

        numbers = extract_numbers(summary)
        unsupported_numbers = [n for n in numbers if n not in source_numbers]

        entities = extract_capitalized_phrases(summary, lang)
        # Deduplicate while preserving order.
        seen = set()
        unique_entities = []
        for e in entities:
            key = normalize_for_match(e)
            if key and key not in seen:
                seen.add(key)
                unique_entities.append(e)
        unsupported_entities = [
            e for e in unique_entities if not phrase_in_text(e, source_norm)
        ]

        total = len(numbers) + len(unique_entities)
        unsupported = len(unsupported_numbers) + len(unsupported_entities)
        score = 1.0 if total == 0 else safe_div(total - unsupported, total)

        return {
            "score": round(score, 4),
            "checked_items": total,
            "unsupported_items": unsupported,
            "numbers_total": len(numbers),
            "unsupported_numbers": unsupported_numbers[:10],
            "entities_total": len(unique_entities),
            "unsupported_entities": unsupported_entities[:10],
            "note": (
                "Proxy metric: checks that numbers and proper nouns in the "
                "summary appear in the source. It cannot detect distorted "
                "relations between facts."
            ),
        }

    def _readability(self, summary: str, lang: str) -> Dict:
        """Readability of the summary: Flesch (English) and LIX (any language)."""
        sentences = sentence_split(summary, lang)
        tokens = tokenize(summary, lang=lang)
        n_sentences = max(1, len(sentences))
        n_words = len(tokens)
        if n_words == 0:
            return {"words_per_sentence": 0.0, "lix": 0.0, "type_token_ratio": 0.0}

        words_per_sentence = safe_div(n_words, n_sentences)
        long_words = sum(1 for t in tokens if len(t) > 6)
        lix = words_per_sentence + 100 * safe_div(long_words, n_words)
        out = {
            "words_per_sentence": round(words_per_sentence, 2),
            "avg_word_length": round(safe_div(sum(len(t) for t in tokens), n_words), 2),
            "lix": round(lix, 2),
            "type_token_ratio": round(safe_div(len(set(tokens)), n_words), 4),
        }
        if lang == "en":
            syllables = sum(count_syllables_en(t) for t in tokens)
            out["flesch_reading_ease"] = round(
                206.835
                - 1.015 * words_per_sentence
                - 84.6 * safe_div(syllables, n_words),
                2,
            )
        return out

    def _length_statistics(self, source: str, summary: str, lang: str = "en") -> Dict:
        """Length statistics, using language-aware sentence splitting."""
        source_sentences = sentence_split(source, lang)
        summary_sentences = sentence_split(summary, lang)
        return {
            "source_chars": len(source),
            "summary_chars": len(summary),
            "source_words": count_words(source),
            "summary_words": count_words(summary),
            "source_sentences": len(source_sentences),
            "summary_sentences": len(summary_sentences),
            "avg_sentence_length_source": round(
                float(np.mean([len(s.split()) for s in source_sentences])), 2
            )
            if source_sentences
            else 0,
            "avg_sentence_length_summary": round(
                float(np.mean([len(s.split()) for s in summary_sentences])), 2
            )
            if summary_sentences
            else 0,
        }

    # ------------------------------------------------------------------ #
    # Composite score
    # ------------------------------------------------------------------ #

    def _compute_overall_quality(self, metrics: Dict) -> Dict:
        """
        Two composites, reported separately and then averaged:

          intrinsic  — content retention, faithfulness, non-redundancy,
                       cohesion (always available)
          reference  — agreement with a human reference summary (only when one
                       was supplied)

        Missing components are dropped and the remaining weights renormalized,
        so a score is never silently inflated by a metric that was not
        computed. `basis` states which families went into the final number, and
        comparing scores across different bases is not meaningful.
        """
        intrinsic_components = {
            "information_density": metrics.get("information_density", {}).get("score"),
            "factual_consistency": metrics.get("factual_consistency", {}).get("score"),
            "source_coverage": metrics.get("coverage_score", {}).get("score"),
            "topic_coverage": metrics.get("topic_coverage", {}).get("score"),
            "non_redundancy": (
                1.0 - metrics["redundancy_score"]["score"]
                if "redundancy_score" in metrics
                else None
            ),
            "lexical_cohesion": metrics.get("coherence_score", {}).get("score"),
        }
        intrinsic = self._weighted(intrinsic_components, INTRINSIC_WEIGHTS)

        reference_components = {}
        if "rouge_scores" in metrics:
            rouge = metrics["rouge_scores"]
            reference_components["rouge_1_f1"] = rouge["rouge1"]["f1"]
            reference_components["rouge_2_f1"] = rouge["rouge2"]["f1"]
            reference_components["rouge_l_f1"] = rouge["rougeL"]["f1"]
        if "meteor_score" in metrics:
            reference_components["meteor"] = metrics["meteor_score"].get("score")
        if "bert_score" in metrics and not metrics["bert_score"].get("error"):
            reference_components["bert_score_f1"] = metrics["bert_score"].get("f1")
        reference = (
            self._weighted(reference_components, REFERENCE_WEIGHTS)
            if reference_components
            else None
        )

        if reference is not None:
            score = 0.5 * reference["score"] + 0.5 * intrinsic["score"]
            basis = "reference+intrinsic"
        else:
            score = intrinsic["score"]
            basis = "intrinsic"

        return {
            "score": round(score, 4),
            "basis": basis,
            "intrinsic_score": intrinsic["score"],
            "reference_score": reference["score"] if reference else None,
            "component_scores": {
                **intrinsic["components"],
                **(reference["components"] if reference else {}),
            },
            "weights": {
                **intrinsic["weights"],
                **(reference["weights"] if reference else {}),
            },
            "note": (
                "Weighted composite. Weights are a stated modelling choice, "
                "not an empirical finding; component scores are reported so "
                "they can be re-weighted. Scores computed on different bases "
                "(with vs. without a reference summary) are not comparable."
            ),
        }

    @staticmethod
    def _weighted(components: Dict[str, Optional[float]], weights: Dict[str, float]) -> Dict:
        """Weighted mean over available components, with renormalized weights."""
        available = {
            k: float(v)
            for k, v in components.items()
            if v is not None and k in weights
        }
        if not available:
            return {"score": 0.0, "components": {}, "weights": {}}
        total_weight = sum(weights[k] for k in available)
        score = sum(available[k] * weights[k] for k in available) / total_weight
        return {
            "score": round(score, 4),
            "components": {k: round(v, 4) for k, v in available.items()},
            "weights": {k: round(weights[k] / total_weight, 4) for k in available},
        }

    # ------------------------------------------------------------------ #
    # Ranking
    # ------------------------------------------------------------------ #

    def _rank_summaries(self, results: Dict) -> Dict:
        """Rank the evaluated summaries by each headline metric."""
        methods = list(results.keys())

        def rank_by(getter) -> List:
            scored = []
            for m in methods:
                value = getter(results[m])
                if value is not None:
                    scored.append((m, value))
            return sorted(scored, key=lambda x: x[1], reverse=True)

        rankings = {
            "by_overall_quality": rank_by(lambda r: r["overall_quality"]["score"]),
            "by_information_density": rank_by(lambda r: r["information_density"]["score"]),
            "by_coherence": rank_by(lambda r: r["coherence_score"]["score"]),
            "by_factual_consistency": rank_by(lambda r: r["factual_consistency"]["score"]),
            "by_source_coverage": rank_by(lambda r: r["coverage_score"]["score"]),
            "by_abstractiveness": rank_by(
                lambda r: r["abstractiveness"]["novel_1gram_ratio"]
            ),
        }
        # ROUGE ranking only exists when a reference summary was provided.
        rouge_ranking = rank_by(
            lambda r: r["rouge_scores"]["rougeL"]["f1"] if "rouge_scores" in r else None
        )
        if rouge_ranking:
            rankings["by_rouge_l"] = rouge_ranking
        return rankings

    # ------------------------------------------------------------------ #
    # Metadata and explanations
    # ------------------------------------------------------------------ #

    @staticmethod
    def metric_metadata() -> Dict[str, Dict]:
        """
        Single source of truth describing every metric: what it measures,
        which direction is better, and where it comes from. The UI renders
        labels and tooltips from this, and it doubles as the methodology table
        for the thesis.
        """
        return {
            "rouge": {
                "label": "ROUGE-1/2/L",
                "family": "reference",
                "direction": "higher",
                "range": "0–1",
                "measures": "N-gram and longest-common-subsequence overlap with a human reference summary.",
                "source": "Lin, 2004",
                "caveat": "Rewards copying the reference's wording; blind to meaning-preserving paraphrase.",
            },
            "meteor": {
                "label": "METEOR",
                "family": "reference",
                "direction": "higher",
                "range": "0–1",
                "measures": "Unigram alignment with stem and synonym matching, plus a fragmentation penalty.",
                "source": "Banerjee & Lavie, 2005",
                "caveat": "Synonym matching uses WordNet and is English-only.",
            },
            "bert_score": {
                "label": "BERTScore",
                "family": "reference",
                "direction": "higher",
                "range": "0–1",
                "measures": "Similarity between contextual embeddings of summary and reference tokens.",
                "source": "Zhang et al., 2020",
                "caveat": "Slow; depends on the underlying encoder's language coverage.",
            },
            "compression_ratio": {
                "label": "Compression ratio",
                "family": "intrinsic",
                "direction": "lower",
                "range": "0–1",
                "measures": "Summary length divided by source length.",
                "source": "—",
                "caveat": "A length statistic, not a quality score.",
            },
            "information_density": {
                "label": "Information density",
                "family": "intrinsic",
                "direction": "higher",
                "range": "0–1",
                "measures": f"Share of the source's top-{N_IMPORTANT_TERMS} TF-IDF terms retained in the summary.",
                "source": "—",
                "caveat": "Term-level; rewards keeping source vocabulary, so extractive methods are favoured.",
            },
            "topic_coverage": {
                "label": "Topic coverage",
                "family": "intrinsic",
                "direction": "higher",
                "range": "0–1",
                "measures": f"Share of the source's top-{N_TOPICS} key phrases present in the summary.",
                "source": "—",
                "caveat": "Key phrases are TF-IDF surface forms, not semantic topics.",
            },
            "source_coverage": {
                "label": "Source coverage",
                "family": "intrinsic",
                "direction": "higher",
                "range": "0–1",
                "measures": f"Share of source sentences with a summary sentence above {COVERAGE_SIM_THRESHOLD} cosine similarity.",
                "source": "—",
                "caveat": "Bounded from above by the compression ratio; compare only at equal summary length.",
            },
            "lexical_cohesion": {
                "label": "Lexical cohesion",
                "family": "intrinsic",
                "direction": "higher",
                "range": "0–1",
                "measures": "Mean similarity between consecutive summary sentences.",
                "source": "proxy for coherence",
                "caveat": "Detects topic jumps only; absolute values are low by nature.",
            },
            "redundancy": {
                "label": "Redundancy",
                "family": "intrinsic",
                "direction": "lower",
                "range": "0–1",
                "measures": f"Share of summary sentence pairs above {REDUNDANCY_SIM_THRESHOLD} similarity, plus distinct-n repetition.",
                "source": "distinct-n: Li et al., 2016",
                "caveat": "Threshold-based; near-duplicates with different wording are missed.",
            },
            "abstractiveness": {
                "label": "Abstractiveness (novel n-grams)",
                "family": "intrinsic",
                "direction": "descriptive",
                "range": "0–1",
                "measures": "Share of summary n-grams that do not occur in the source.",
                "source": "See et al., 2017",
                "caveat": "Descriptive, not better/worse: novelty is both abstraction and hallucination.",
            },
            "extractive_fragments": {
                "label": "Extractive coverage & density",
                "family": "intrinsic",
                "direction": "descriptive",
                "range": "coverage 0–1; density ≥ 0",
                "measures": "How much of the summary is copied from the source, and how long the copied runs are.",
                "source": "Grusky et al., 2018",
                "caveat": "Descriptive measure of extraction strategy.",
            },
            "factual_consistency": {
                "label": "Factual consistency (proxy)",
                "family": "intrinsic",
                "direction": "higher",
                "range": "0–1",
                "measures": "Share of numbers and proper nouns in the summary that appear in the source.",
                "source": "proxy, in the spirit of Kryscinski et al., 2020",
                "caveat": "Surface check; cannot detect distorted relations between facts.",
            },
            "readability": {
                "label": "Readability",
                "family": "intrinsic",
                "direction": "descriptive",
                "range": "Flesch 0–100; LIX ≥ 0",
                "measures": "Sentence and word length based readability (Flesch for English, LIX for any language).",
                "source": "Flesch, 1948; Björnsson, 1968",
                "caveat": "Surface formulas; LIX penalizes agglutinative languages such as Turkish.",
            },
            "overall_quality": {
                "label": "Overall quality",
                "family": "composite",
                "direction": "higher",
                "range": "0–1",
                "measures": "Weighted composite of the intrinsic metrics and, when a reference exists, the reference metrics.",
                "source": "this work",
                "caveat": "Weights are a stated choice; only comparable between systems evaluated on the same basis.",
            },
        }

    def _generate_explanations(self, metrics: Dict) -> Dict:
        """Plain-language explanation of each score, for the UI and the paper."""
        explanations = {}

        if "rouge_scores" in metrics:
            r1 = metrics["rouge_scores"]["rouge1"]
            r2 = metrics["rouge_scores"]["rouge2"]
            rl = metrics["rouge_scores"]["rougeL"]
            explanations["rouge"] = (
                f"Measured against the human reference summary. ROUGE-1 F1 is {r1['f1']:.1%} "
                f"({r1['recall']:.0%} of reference words recalled, {r1['precision']:.0%} of summary words relevant). "
                f"ROUGE-2 F1 is {r2['f1']:.1%}, indicating {'good' if r2['f1'] > 0.15 else 'limited'} "
                f"phrase-level agreement. ROUGE-L F1 is {rl['f1']:.1%}."
            )
        else:
            explanations["rouge"] = (
                "No reference summary was supplied, so ROUGE, METEOR and BERTScore are not "
                "reported. Scoring a summary against its own source text is not ROUGE and is "
                "shown separately as the copy rate."
            )

        if "source_overlap" in metrics:
            so = metrics["source_overlap"]
            explanations["source_overlap"] = (
                f"{so['unigram_copy_rate']:.0%} of the summary's words and "
                f"{so['bigram_copy_rate']:.0%} of its word pairs are copied verbatim from the source. "
                f"{'Mostly extraction.' if so['unigram_copy_rate'] > 0.9 else 'Substantial rewriting.'}"
            )

        if "compression_ratio" in metrics:
            cr = metrics["compression_ratio"]
            explanations["compression_ratio"] = (
                f"The summary is {cr['summary_words']} words against the source's {cr['source_words']} "
                f"— a {cr['reduction_percentage']}% reduction. "
                f"{'Typical for summarization (70–90%).' if cr['reduction_percentage'] > 60 else 'Less compression than is usual.'}"
            )

        if "information_density" in metrics:
            info = metrics["information_density"]
            explanations["information_density"] = (
                f"{info['important_terms_in_summary']} of the source's {info['important_terms_total']} "
                f"highest-TF-IDF terms survive in the summary ({info['score']:.0%}). "
                f"{'Key terminology is retained.' if info['score'] > 0.6 else 'Key concepts were dropped.'}"
            )

        if "abstractiveness" in metrics:
            ab = metrics["abstractiveness"]
            explanations["abstractiveness"] = (
                f"{ab['novel_1gram_ratio']:.0%} of words and {ab['novel_2gram_ratio']:.0%} of word pairs "
                f"in the summary do not appear in the source. "
                + (
                    "Effectively extractive — sentences are reused verbatim."
                    if ab["novel_1gram_ratio"] < 0.05
                    else "Genuine rewriting; novel content is also where hallucination appears, so check faithfulness."
                )
            )

        if "extractive_fragments" in metrics:
            ef = metrics["extractive_fragments"]
            explanations["extractive_fragments"] = (
                f"Extractive coverage {ef['coverage']:.0%} with density {ef['density']:.1f} "
                f"(longest copied run: {ef['longest_fragment']} tokens). Density near 1 means word-level "
                f"reuse; high density means whole sentences were copied."
            )

        if "factual_consistency" in metrics:
            fc = metrics["factual_consistency"]
            if fc["checked_items"] == 0:
                explanations["factual_consistency"] = (
                    "No numbers or proper nouns were found in the summary, so faithfulness could not be checked."
                )
            else:
                detail = ""
                if fc["unsupported_numbers"]:
                    detail += f" Unsupported numbers: {', '.join(fc['unsupported_numbers'][:5])}."
                if fc["unsupported_entities"]:
                    detail += f" Unsupported names: {', '.join(fc['unsupported_entities'][:5])}."
                explanations["factual_consistency"] = (
                    f"{fc['checked_items'] - fc['unsupported_items']} of {fc['checked_items']} numbers and "
                    f"proper nouns in the summary are traceable to the source ({fc['score']:.0%})."
                    + (detail if detail else " Nothing unsupported was detected.")
                )

        if "redundancy_score" in metrics:
            red = metrics["redundancy_score"]
            if red["num_sentences"] <= 1:
                explanations["redundancy"] = (
                    f"The summary is a single sentence, so sentence-level redundancy does not apply. "
                    f"Distinct-2 is {red['distinct_2']:.0%}."
                )
            else:
                explanations["redundancy"] = (
                    f"{red['redundant_pairs']} of the sentence pairs among {red['num_sentences']} sentences "
                    f"exceed {REDUNDANCY_SIM_THRESHOLD:.0%} similarity ({red['score']:.0%}). "
                    f"Distinct-2 is {red['distinct_2']:.0%} and {red['repeated_4gram_rate']:.0%} of 4-grams repeat"
                    + (" — a sign of decoder looping." if red["repeated_4gram_rate"] > 0.1 else ".")
                )

        if "coherence_score" in metrics:
            coh = metrics["coherence_score"]
            if coh.get("single_sentence"):
                explanations["coherence"] = "Single-sentence summary: cohesion between sentences cannot be measured."
            else:
                explanations["coherence"] = (
                    f"Consecutive sentences share {coh['avg_transition_similarity']:.1%} average similarity "
                    f"(weakest {coh['min_transition']:.1%}, strongest {coh['max_transition']:.1%}). "
                    + (
                        "Sentences follow each other topically."
                        if coh["score"] > 0.15
                        else "Low cohesion: the summary jumps between topics — typical of sentence extraction and of merged chunk summaries."
                    )
                )

        if "coverage_score" in metrics:
            cov = metrics["coverage_score"]
            explanations["coverage"] = (
                f"{cov['covered_sentences']} of {cov['total_sentences']} source sentences are represented "
                f"in the summary above {cov['threshold']:.0%} similarity ({cov['score']:.0%}). "
                f"{'Most of the document is represented.' if cov['score'] > 0.5 else 'The summary represents specific sections rather than the whole document.'}"
            )

        if "topic_coverage" in metrics:
            tc = metrics["topic_coverage"]
            covered = ", ".join(tc["covered_topics"][:5]) if tc["covered_topics"] else "none"
            missing = ", ".join(tc["missing_topics"][:5]) if tc["missing_topics"] else "none"
            explanations["topic_coverage"] = (
                f"{len(tc['covered_topics'])}/{tc['total_topics']} key phrases present. "
                f"Covered: [{covered}]. Missing: [{missing}]."
            )

        if "readability" in metrics:
            rd = metrics["readability"]
            flesch = rd.get("flesch_reading_ease")
            explanations["readability"] = (
                f"{rd['words_per_sentence']} words per sentence, LIX {rd['lix']}, "
                f"type/token ratio {rd['type_token_ratio']:.2f}"
                + (f", Flesch Reading Ease {flesch}." if flesch is not None else ".")
            )

        if "overall_quality" in metrics:
            oq = metrics["overall_quality"]
            parts = [
                f"{k.replace('_', ' ')} {v:.0%}×{oq['weights'].get(k, 0):.2f}"
                for k, v in oq.get("component_scores", {}).items()
            ]
            basis = (
                "reference metrics and intrinsic metrics, equally weighted"
                if oq["basis"] == "reference+intrinsic"
                else "intrinsic metrics only (no reference summary was supplied)"
            )
            explanations["overall_quality"] = (
                f"Overall quality {oq['score']:.1%}, computed from {basis}. Components: {'; '.join(parts)}. "
                f"The weights are a stated modelling choice, so treat this number as a summary of the "
                f"individual metrics rather than as ground truth."
            )

        return explanations
