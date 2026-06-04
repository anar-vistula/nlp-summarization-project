"""
Summary Evaluation Metrics Module (Multilingual).
Provides comprehensive evaluation of generated summaries using multiple metrics:
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BERTScore (semantic similarity)
- Compression Ratio
- Information Density
- Redundancy Score
- Coherence Score
- Coverage Score
"""

import numpy as np
from typing import Dict, Optional, List
from rouge_score import rouge_scorer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.preprocessing.text_preprocessor import TextPreprocessor

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class SummaryEvaluator:
    """Comprehensive multilingual summary evaluation."""

    def __init__(self, use_bertscore: bool = False):
        """
        Args:
            use_bertscore: Whether to compute BERTScore (slower but more accurate).
        """
        self.use_bertscore = use_bertscore
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.preprocessor = TextPreprocessor()

    def evaluate(
        self,
        source_text: str,
        summary: str,
        reference_summary: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a summary comprehensively.

        Args:
            source_text: Original input text
            summary: Generated summary to evaluate
            reference_summary: Optional human-written reference summary
            lang: Language code (auto-detected if None)

        Returns:
            Dictionary containing all computed metrics
        """
        # Detect language if not provided
        if lang is None:
            lang = self.preprocessor.detect_language(source_text)

        stopwords = self.preprocessor.get_stopwords(lang)
        sw_list = list(stopwords) if stopwords else None

        metrics = {}
        metrics["language"] = lang

        # --- Content Metrics (reference-free) ---
        metrics["compression_ratio"] = self._compression_ratio(source_text, summary)
        metrics["information_density"] = self._information_density(source_text, summary, sw_list)
        metrics["redundancy_score"] = self._redundancy_score(summary, sw_list, lang)
        metrics["coherence_score"] = self._coherence_score(summary, sw_list, lang)
        metrics["coverage_score"] = self._coverage_score(source_text, summary, sw_list, lang)
        metrics["length_stats"] = self._length_statistics(source_text, summary)
        metrics["topic_coverage"] = self._topic_coverage(source_text, summary, sw_list)

        # --- Reference-based Metrics (when reference is available) ---
        if reference_summary:
            metrics["rouge_scores"] = self._compute_rouge(reference_summary, summary)
        else:
            # Use source text as pseudo-reference for ROUGE
            metrics["rouge_scores"] = self._compute_rouge(source_text, summary)

        # --- BERTScore (optional, computationally expensive) ---
        if self.use_bertscore:
            metrics["bert_score"] = self._compute_bertscore(source_text, summary)

        # --- Overall Quality Score (weighted combination) ---
        metrics["overall_quality"] = self._compute_overall_quality(metrics)

        # --- Human-readable explanations for each metric ---
        metrics["explanations"] = self._generate_explanations(metrics)

        return metrics

    def _compute_rouge(self, reference: str, summary: str) -> Dict:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
        scores = self.rouge_scorer.score(reference, summary)
        return {
            "rouge1": {
                "precision": round(scores["rouge1"].precision, 4),
                "recall": round(scores["rouge1"].recall, 4),
                "f1": round(scores["rouge1"].fmeasure, 4),
            },
            "rouge2": {
                "precision": round(scores["rouge2"].precision, 4),
                "recall": round(scores["rouge2"].recall, 4),
                "f1": round(scores["rouge2"].fmeasure, 4),
            },
            "rougeL": {
                "precision": round(scores["rougeL"].precision, 4),
                "recall": round(scores["rougeL"].recall, 4),
                "f1": round(scores["rougeL"].fmeasure, 4),
            },
        }

    def _compression_ratio(self, source: str, summary: str) -> Dict:
        """
        Compute compression ratio.
        Lower ratio = more compression. Ideal range: 0.1 - 0.3
        """
        source_words = len(source.split())
        summary_words = len(summary.split())
        ratio = summary_words / source_words if source_words > 0 else 0
        return {
            "ratio": round(ratio, 4),
            "source_words": source_words,
            "summary_words": summary_words,
            "reduction_percentage": round((1 - ratio) * 100, 2),
        }

    def _information_density(self, source: str, summary: str, stopwords: Optional[list] = None) -> Dict:
        """
        Measure how much unique information is packed into the summary.
        Uses TF-IDF to identify important terms and checks their presence in summary.
        """
        try:
            vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=50)
            vectorizer.fit([source])
            important_terms = vectorizer.get_feature_names_out()

            summary_lower = summary.lower()
            terms_in_summary = sum(1 for term in important_terms if term in summary_lower)
            density = terms_in_summary / len(important_terms) if important_terms.size > 0 else 0

            return {
                "score": round(density, 4),
                "important_terms_total": len(important_terms),
                "important_terms_in_summary": terms_in_summary,
            }
        except ValueError:
            return {"score": 0.0, "important_terms_total": 0, "important_terms_in_summary": 0}

    def _redundancy_score(self, summary: str, stopwords: Optional[list] = None, lang: str = "en") -> Dict:
        """
        Detect redundancy in the summary.
        Lower score = less redundancy = better.
        Measures sentence-level similarity within the summary.
        """
        from backend.preprocessing.text_preprocessor import PUNKT_LANG_MAP
        punkt_lang = PUNKT_LANG_MAP.get(lang, "english")
        try:
            sentences = nltk.sent_tokenize(summary, language=punkt_lang)
        except LookupError:
            sentences = nltk.sent_tokenize(summary)
        if len(sentences) <= 1:
            return {"score": 0.0, "num_sentences": len(sentences), "redundant_pairs": 0}

        try:
            vectorizer = TfidfVectorizer(stop_words=stopwords)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sim_matrix = cosine_similarity(tfidf_matrix)
            np.fill_diagonal(sim_matrix, 0)

            redundant_pairs = int(np.sum(sim_matrix > 0.5) / 2)
            total_pairs = len(sentences) * (len(sentences) - 1) / 2
            redundancy = redundant_pairs / total_pairs if total_pairs > 0 else 0

            return {
                "score": round(redundancy, 4),
                "num_sentences": len(sentences),
                "redundant_pairs": redundant_pairs,
                "max_similarity": round(float(sim_matrix.max()), 4) if sim_matrix.size > 0 else 0,
            }
        except ValueError:
            return {"score": 0.0, "num_sentences": len(sentences), "redundant_pairs": 0}

    def _coherence_score(self, summary: str, stopwords: Optional[list] = None, lang: str = "en") -> Dict:
        """
        Measure coherence of the summary.
        Checks how well consecutive sentences relate to each other.
        Higher score = better coherence.
        """
        from backend.preprocessing.text_preprocessor import PUNKT_LANG_MAP
        punkt_lang = PUNKT_LANG_MAP.get(lang, "english")
        try:
            sentences = nltk.sent_tokenize(summary, language=punkt_lang)
        except LookupError:
            sentences = nltk.sent_tokenize(summary)
        if len(sentences) <= 1:
            return {"score": 1.0, "num_transitions": 0, "avg_transition_similarity": 1.0}

        try:
            vectorizer = TfidfVectorizer(stop_words=stopwords)
            tfidf_matrix = vectorizer.fit_transform(sentences)

            transition_scores = []
            for i in range(len(sentences) - 1):
                sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0][0]
                transition_scores.append(float(sim))

            avg_coherence = np.mean(transition_scores) if transition_scores else 0

            return {
                "score": round(float(avg_coherence), 4),
                "num_transitions": len(transition_scores),
                "avg_transition_similarity": round(float(avg_coherence), 4),
                "min_transition": round(float(min(transition_scores)), 4) if transition_scores else 0,
                "max_transition": round(float(max(transition_scores)), 4) if transition_scores else 0,
            }
        except ValueError:
            return {"score": 0.0, "num_transitions": 0, "avg_transition_similarity": 0.0}

    def _coverage_score(self, source: str, summary: str, stopwords: Optional[list] = None, lang: str = "en") -> Dict:
        """
        Measure how well the summary covers the source content.
        Uses sentence-level coverage analysis.
        """
        from backend.preprocessing.text_preprocessor import PUNKT_LANG_MAP
        punkt_lang = PUNKT_LANG_MAP.get(lang, "english")
        try:
            source_sentences = nltk.sent_tokenize(source, language=punkt_lang)
            summary_sentences = nltk.sent_tokenize(summary, language=punkt_lang)
        except LookupError:
            source_sentences = nltk.sent_tokenize(source)
            summary_sentences = nltk.sent_tokenize(summary)

        if not source_sentences or not summary_sentences:
            return {"score": 0.0, "covered_sentences": 0, "total_sentences": 0}

        try:
            all_sentences = source_sentences + summary_sentences
            vectorizer = TfidfVectorizer(stop_words=stopwords)
            tfidf_matrix = vectorizer.fit_transform(all_sentences)

            source_vectors = tfidf_matrix[:len(source_sentences)]
            summary_vectors = tfidf_matrix[len(source_sentences):]

            sim_matrix = cosine_similarity(source_vectors, summary_vectors)
            max_similarities = sim_matrix.max(axis=1)

            covered = int(np.sum(max_similarities > 0.3))
            coverage = covered / len(source_sentences)

            return {
                "score": round(coverage, 4),
                "covered_sentences": covered,
                "total_sentences": len(source_sentences),
                "avg_coverage": round(float(max_similarities.mean()), 4),
            }
        except ValueError:
            return {"score": 0.0, "covered_sentences": 0, "total_sentences": len(source_sentences)}

    def _topic_coverage(self, source: str, summary: str, stopwords: Optional[list] = None) -> Dict:
        """
        Analyze how well the summary covers the main topics of the source.
        Extracts key topics using TF-IDF and checks their presence.
        """
        try:
            vectorizer = TfidfVectorizer(
                stop_words=stopwords, max_features=20, ngram_range=(1, 2)
            )
            vectorizer.fit([source])
            topics = vectorizer.get_feature_names_out().tolist()

            summary_lower = summary.lower()
            covered_topics = [t for t in topics if t in summary_lower]
            missing_topics = [t for t in topics if t not in summary_lower]

            coverage = len(covered_topics) / len(topics) if topics else 0

            return {
                "score": round(coverage, 4),
                "total_topics": len(topics),
                "covered_topics": covered_topics,
                "missing_topics": missing_topics,
            }
        except ValueError:
            return {"score": 0.0, "total_topics": 0, "covered_topics": [], "missing_topics": []}

    def _length_statistics(self, source: str, summary: str) -> Dict:
        """Compute detailed length statistics."""
        source_sentences = nltk.sent_tokenize(source)
        summary_sentences = nltk.sent_tokenize(summary)

        return {
            "source_chars": len(source),
            "summary_chars": len(summary),
            "source_words": len(source.split()),
            "summary_words": len(summary.split()),
            "source_sentences": len(source_sentences),
            "summary_sentences": len(summary_sentences),
            "avg_sentence_length_source": round(
                np.mean([len(s.split()) for s in source_sentences]), 2
            ) if source_sentences else 0,
            "avg_sentence_length_summary": round(
                np.mean([len(s.split()) for s in summary_sentences]), 2
            ) if summary_sentences else 0,
        }

    def _compute_bertscore(self, source: str, summary: str) -> Dict:
        """Compute BERTScore for semantic similarity."""
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(
                [summary], [source], lang="en", verbose=False
            )
            return {
                "precision": round(float(P[0]), 4),
                "recall": round(float(R[0]), 4),
                "f1": round(float(F1[0]), 4),
            }
        except Exception as e:
            return {"error": str(e), "precision": 0, "recall": 0, "f1": 0}

    def _compute_overall_quality(self, metrics: Dict) -> Dict:
        """
        Compute a weighted overall quality score combining all metrics.
        Score ranges from 0 to 1, where 1 is best.
        """
        scores = {}
        weights = {}

        # ROUGE-L F1 (most important for content quality)
        if "rouge_scores" in metrics:
            scores["rouge_l_f1"] = metrics["rouge_scores"]["rougeL"]["f1"]
            weights["rouge_l_f1"] = 0.25

            scores["rouge_1_f1"] = metrics["rouge_scores"]["rouge1"]["f1"]
            weights["rouge_1_f1"] = 0.15

            scores["rouge_2_f1"] = metrics["rouge_scores"]["rouge2"]["f1"]
            weights["rouge_2_f1"] = 0.10

        # Information density
        if "information_density" in metrics:
            scores["info_density"] = metrics["information_density"]["score"]
            weights["info_density"] = 0.15

        # Low redundancy (invert: lower redundancy = higher score)
        if "redundancy_score" in metrics:
            scores["non_redundancy"] = 1.0 - metrics["redundancy_score"]["score"]
            weights["non_redundancy"] = 0.10

        # Coherence
        if "coherence_score" in metrics:
            scores["coherence"] = metrics["coherence_score"]["score"]
            weights["coherence"] = 0.10

        # Coverage
        if "coverage_score" in metrics:
            scores["coverage"] = metrics["coverage_score"]["score"]
            weights["coverage"] = 0.10

        # Topic coverage
        if "topic_coverage" in metrics:
            scores["topic_coverage"] = metrics["topic_coverage"]["score"]
            weights["topic_coverage"] = 0.05

        # Compute weighted average
        if not weights:
            return {"score": 0.0, "component_scores": {}}

        total_weight = sum(weights.values())
        weighted_sum = sum(scores[k] * weights[k] for k in scores)
        overall = weighted_sum / total_weight if total_weight > 0 else 0

        return {
            "score": round(overall, 4),
            "component_scores": {k: round(v, 4) for k, v in scores.items()},
            "weights": weights,
        }

    def _generate_explanations(self, metrics: Dict) -> Dict:
        """
        Generate human-readable explanations for why each metric has its value.
        """
        explanations = {}

        # ROUGE explanation
        if "rouge_scores" in metrics:
            r1 = metrics["rouge_scores"]["rouge1"]
            r2 = metrics["rouge_scores"]["rouge2"]
            rl = metrics["rouge_scores"]["rougeL"]
            explanations["rouge"] = (
                f"ROUGE-1 F1 is {r1['f1']:.2%} — meaning {r1['recall']:.0%} of the words in the reference "
                f"appear in the summary (recall), and {r1['precision']:.0%} of the summary words are relevant (precision). "
                f"ROUGE-2 F1 is {r2['f1']:.2%} — only {r2['recall']:.0%} of two-word phrases from the reference were preserved, "
                f"indicating {'good' if r2['f1'] > 0.15 else 'limited'} phrase-level fidelity. "
                f"ROUGE-L F1 is {rl['f1']:.2%} — the longest matching sequence covers "
                f"{'a significant' if rl['f1'] > 0.3 else 'a small'} portion of the text structure."
            )

        # Compression ratio explanation
        if "compression_ratio" in metrics:
            cr = metrics["compression_ratio"]
            explanations["compression_ratio"] = (
                f"The summary is {cr['summary_words']} words compared to the source's {cr['source_words']} words — "
                f"a {cr['reduction_percentage']}% reduction. "
                f"{'This is a good compression level (70-90% reduction is typical).' if cr['reduction_percentage'] > 60 else 'The summary could be more concise.'}"
            )

        # Information density explanation
        if "information_density" in metrics:
            info = metrics["information_density"]
            explanations["information_density"] = (
                f"Of the {info['important_terms_total']} most important terms identified in the source (via TF-IDF), "
                f"{info['important_terms_in_summary']} appear in the summary ({info['score']:.0%}). "
                f"{'The summary retains most key terminology.' if info['score'] > 0.6 else 'Some important terms were lost — the summary may miss key concepts.'}"
            )

        # Redundancy explanation
        if "redundancy_score" in metrics:
            red = metrics["redundancy_score"]
            if red["num_sentences"] <= 1:
                explanations["redundancy"] = "The summary has only one sentence, so redundancy is not applicable."
            else:
                redundancy_detail = f"Redundancy score is {red['score']:.0%} — some sentences convey overlapping information."
                if red["score"] < 0.1:
                    redundancy_detail = "No significant repetition detected — each sentence adds new information."
                explanations["redundancy"] = (
                    f"Out of {red['num_sentences']} sentences in the summary, {red['redundant_pairs']} pairs "
                    f"have high similarity (>50% cosine similarity). "
                    f"{redundancy_detail}"
                )

        # Coherence explanation
        if "coherence_score" in metrics:
            coh = metrics["coherence_score"]
            if coh["num_transitions"] == 0:
                explanations["coherence"] = "The summary has only one sentence, so coherence between sentences cannot be measured."
            else:
                explanations["coherence"] = (
                    f"Average similarity between consecutive sentences is {coh['avg_transition_similarity']:.2%}. "
                    f"The weakest transition has {coh['min_transition']:.2%} similarity and the strongest has {coh['max_transition']:.2%}. "
                    f"{'Sentences flow well together topically.' if coh['score'] > 0.15 else 'Low coherence suggests the summary jumps between unrelated topics — common in extractive methods that pick sentences from different parts of the document.'}"
                )

        # Coverage explanation
        if "coverage_score" in metrics:
            cov = metrics["coverage_score"]
            explanations["coverage"] = (
                f"{cov['covered_sentences']} out of {cov['total_sentences']} source sentences "
                f"have at least one similar sentence in the summary ({cov['score']:.0%} coverage). "
                f"{'The summary represents most of the source content.' if cov['score'] > 0.5 else 'Large portions of the source are not represented — the summary focuses on specific sections.'}"
            )

        # Topic coverage explanation
        if "topic_coverage" in metrics:
            tc = metrics["topic_coverage"]
            covered = ", ".join(tc["covered_topics"][:5]) if tc["covered_topics"] else "none"
            missing = ", ".join(tc["missing_topics"][:5]) if tc["missing_topics"] else "none"
            explanations["topic_coverage"] = (
                f"{len(tc['covered_topics'])}/{tc['total_topics']} key topics are present in the summary. "
                f"Covered: [{covered}]. "
                f"Missing: [{missing}]. "
                f"{'Good topic retention.' if tc['score'] > 0.6 else 'Several important topics are absent from the summary.'}"
            )

        # Overall quality explanation
        if "overall_quality" in metrics:
            oq = metrics["overall_quality"]
            components = oq.get("component_scores", {})
            weights = oq.get("weights", {})
            breakdown_parts = []
            for key, score in components.items():
                weight = weights.get(key, 0)
                contribution = score * weight
                breakdown_parts.append(f"{key.replace('_', ' ')} contributed {contribution:.2%} (score {score:.2%} × weight {weight})")

            explanations["overall_quality"] = (
                f"Overall quality is {oq['score']:.2%}, computed as a weighted average of all metrics. "
                f"Breakdown: {'; '.join(breakdown_parts)}."
            )

        return explanations

    def compare_summaries(
        self,
        source_text: str,
        summaries: Dict[str, str],
        reference_summary: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Dict:
        """
        Compare multiple summaries against each other.

        Args:
            source_text: Original text
            summaries: Dict mapping method name to summary text
            reference_summary: Optional reference summary
            lang: Language code

        Returns:
            Comparative evaluation results
        """
        results = {}
        for method_name, summary in summaries.items():
            results[method_name] = self.evaluate(source_text, summary, reference_summary, lang=lang)

        # Add ranking
        rankings = self._rank_summaries(results)
        return {
            "individual_results": results,
            "rankings": rankings,
        }

    def _rank_summaries(self, results: Dict) -> Dict:
        """Rank summaries by different criteria."""
        rankings = {}
        methods = list(results.keys())

        # Rank by overall quality
        quality_scores = {m: results[m]["overall_quality"]["score"] for m in methods}
        rankings["by_overall_quality"] = sorted(
            quality_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Rank by ROUGE-L
        rouge_scores = {m: results[m]["rouge_scores"]["rougeL"]["f1"] for m in methods}
        rankings["by_rouge_l"] = sorted(
            rouge_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Rank by information density
        density_scores = {m: results[m]["information_density"]["score"] for m in methods}
        rankings["by_information_density"] = sorted(
            density_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Rank by coherence
        coherence_scores = {m: results[m]["coherence_score"]["score"] for m in methods}
        rankings["by_coherence"] = sorted(
            coherence_scores.items(), key=lambda x: x[1], reverse=True
        )

        return rankings
