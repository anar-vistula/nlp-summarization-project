"""
Correctness tests for the evaluation metrics.

Each test pins a value that can be verified by hand, so a regression in a
metric shows up as a failing number rather than as a plausible-looking score.
Run with:  ./venv/bin/python -m pytest tests -q
"""

import math

from backend.evaluation.metrics import SummaryEvaluator
from backend.evaluation.statistics import (
    bootstrap_mean_ci,
    cohens_d_paired,
    holm_bonferroni,
    paired_bootstrap_test,
)
from backend.evaluation.text_utils import (
    extract_capitalized_phrases,
    extract_numbers,
    ngrams,
    term_in_tokens,
    tokenize,
)

EN_SOURCE = (
    "Climate change refers to long-term shifts in temperatures. "
    "Burning fossil fuels generates greenhouse gas emissions. "
    "The Earth is now about 1.1 degrees Celsius warmer than in the late 1800s. "
    "The Paris Agreement aims to limit warming to 1.5 degrees Celsius. "
    "Renewable energy sources are becoming cheaper every year."
)

TR_SOURCE = (
    "Yapay zeka bilgisayar bilimlerinde önemli bir teknolojiye dönüştü. "
    "Büyük dil modelleri doğal dil üretiminde olağanüstü yetenekler gösterdi. "
    "Sağlık sektörü yapay zekadan büyük ölçüde faydalanacaktır. "
    "İstanbul'daki araştırma merkezleri bu alanda çalışmalar yürütüyor."
)


# ---------------------------------------------------------------- tokenizer

def test_tokenizer_preserves_turkish_characters():
    tokens = tokenize("Öğretmen İstanbul'un şehrinde çalışıyor.", lang="tr")
    assert tokens == ["öğretmen", "istanbul'un", "şehrinde", "çalışıyor"]


def test_tokenizer_stems_english_only():
    assert tokenize("emissions", lang="en", stem=True) == ["emiss"]
    # Turkish must not be run through the English Porter stemmer.
    assert tokenize("emissions", lang="tr", stem=True) == ["emissions"]


def test_ngrams_and_term_matching():
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    tokens = {"climates", "hükümetin", "ai"}
    assert term_in_tokens("climate", tokens)      # prefix match, inflected form
    assert term_in_tokens("hükümet", tokens)      # Turkish agglutination
    assert term_in_tokens("ai", tokens)           # exact short token
    assert not term_in_tokens("said", tokens)     # no false substring hit


def test_number_and_entity_extraction():
    assert extract_numbers("It rose 1.5 degrees and 68 per cent, up from 1,200.") == [
        "1.5",
        "68",
        "1200",
    ]
    phrases = extract_capitalized_phrases("The Paris Agreement was signed. Mustafa Kemal spoke.")
    assert "Paris Agreement" in phrases
    # Sentence-initial capitals are grammatical, not proper nouns.
    assert "The" not in phrases and "Mustafa" not in " ".join(p for p in phrases if p == "The")


# ------------------------------------------------------------------- ROUGE

def test_rouge_requires_reference_and_is_absent_otherwise():
    ev = SummaryEvaluator()
    without = ev.evaluate(EN_SOURCE, "Climate change is driven by fossil fuels.")
    assert "rouge_scores" not in without
    assert without["reference_available"] is False
    # The source-overlap replacement is always present and clearly named.
    assert "unigram_copy_rate" in without["source_overlap"]

    with_ref = ev.evaluate(
        EN_SOURCE,
        "Climate change is driven by fossil fuels.",
        reference_summary="Climate change is caused by burning fossil fuels.",
    )
    assert with_ref["reference_available"] is True
    assert 0.0 < with_ref["rouge_scores"]["rouge1"]["f1"] <= 1.0


def test_rouge_identical_text_is_perfect():
    ev = SummaryEvaluator()
    text = "Renewable energy sources are becoming cheaper every year."
    scores = ev._compute_rouge(text, text, lang="en")
    assert scores["rouge1"]["f1"] == 1.0
    assert scores["rouge2"]["f1"] == 1.0
    assert scores["rougeL"]["f1"] == 1.0


def test_rouge_is_not_destroyed_by_turkish_characters():
    """The default rouge_score tokenizer strips non-ASCII; ours must not."""
    ev = SummaryEvaluator()
    summary = "Büyük dil modelleri olağanüstü yetenekler gösterdi."
    scores = ev._compute_rouge(summary, summary, lang="tr")
    assert scores["rouge1"]["f1"] == 1.0

    # A summary that shares no words with the reference must score 0, not a
    # false positive produced by mangled tokens.
    other = ev._compute_rouge("Sağlık sektörü faydalanacaktır.", summary, lang="tr")
    assert other["rouge1"]["f1"] == 0.0


# ---------------------------------------------------------- intrinsic core

def test_extractive_summary_has_no_novel_ngrams():
    ev = SummaryEvaluator()
    copied = "Burning fossil fuels generates greenhouse gas emissions."
    metrics = ev.evaluate(EN_SOURCE, copied)
    assert metrics["abstractiveness"]["novel_1gram_ratio"] == 0.0
    assert metrics["abstractiveness"]["novel_2gram_ratio"] == 0.0
    # Fully copied: coverage 1.0, and density equals the fragment length.
    assert metrics["extractive_fragments"]["coverage"] == 1.0
    assert metrics["extractive_fragments"]["longest_fragment"] == 7
    assert metrics["source_overlap"]["unigram_copy_rate"] == 1.0


def test_abstractive_summary_has_novel_ngrams():
    ev = SummaryEvaluator()
    paraphrase = "Humanity must urgently abandon coal to protect the planet."
    metrics = ev.evaluate(EN_SOURCE, paraphrase)
    assert metrics["abstractiveness"]["novel_1gram_ratio"] > 0.5
    assert metrics["extractive_fragments"]["coverage"] < 0.5


def test_extractive_fragments_match_grusky_definition():
    """coverage = copied tokens / summary tokens; density = mean squared run."""
    ev = SummaryEvaluator()
    source = tokenize("a b c d e f", lang="en")
    summary = tokenize("a b c x y", lang="en")
    frag = ev._extractive_fragments(source, summary)
    # One fragment of length 3 ("a b c") out of 5 summary tokens.
    assert frag["coverage"] == 0.6
    assert frag["density"] == round(9 / 5, 4)
    assert frag["longest_fragment"] == 3
    assert frag["num_fragments"] == 1


def test_factual_consistency_flags_invented_facts():
    ev = SummaryEvaluator()
    faithful = ev.evaluate(
        EN_SOURCE, "The Paris Agreement targets 1.5 degrees Celsius of warming."
    )["factual_consistency"]
    assert faithful["score"] == 1.0
    assert faithful["unsupported_numbers"] == []

    hallucinated = ev.evaluate(
        EN_SOURCE, "The Kyoto Protocol of 1997 limited warming to 9.9 degrees."
    )["factual_consistency"]
    assert hallucinated["score"] < 1.0
    assert "9.9" in hallucinated["unsupported_numbers"]
    assert any("Kyoto" in e for e in hallucinated["unsupported_entities"])


def test_redundancy_detects_repetition():
    ev = SummaryEvaluator()
    repeated = (
        "Renewable energy is cheaper every year. Renewable energy is cheaper every year."
    )
    metrics = ev.evaluate(EN_SOURCE, repeated)["redundancy_score"]
    assert metrics["score"] == 1.0          # the only sentence pair is redundant
    assert metrics["distinct_1"] < 0.6      # every token occurs twice
    # 12 tokens -> 9 four-grams, 6 of them unique: a third repeat.
    assert metrics["repeated_4gram_rate"] == round(1 - 6 / 9, 4)

    varied = ev.evaluate(
        EN_SOURCE,
        "Climate change shifts temperatures. The Paris Agreement targets 1.5 degrees.",
    )["redundancy_score"]
    assert varied["score"] == 0.0


def test_compression_and_density_directions():
    ev = SummaryEvaluator()
    short = ev.evaluate(EN_SOURCE, "Fossil fuels cause warming.")
    long_summary = ev.evaluate(EN_SOURCE, EN_SOURCE)
    assert short["compression_ratio"]["ratio"] < long_summary["compression_ratio"]["ratio"]
    # Repeating the source retains all of its key terms.
    assert long_summary["information_density"]["score"] == 1.0
    assert short["information_density"]["score"] < 0.5


def test_turkish_document_is_evaluated_end_to_end():
    ev = SummaryEvaluator()
    metrics = ev.evaluate(
        TR_SOURCE,
        "Büyük dil modelleri doğal dil üretiminde olağanüstü yetenekler gösterdi.",
        reference_summary="Yapay zeka ve dil modelleri hızla gelişiyor.",
        lang="tr",
    )
    assert metrics["language"] == "tr"
    assert metrics["information_density"]["score"] > 0
    assert metrics["abstractiveness"]["novel_1gram_ratio"] == 0.0  # copied sentence
    assert 0.0 <= metrics["overall_quality"]["score"] <= 1.0


# -------------------------------------------------------------- composites

def test_overall_quality_basis_and_renormalization():
    ev = SummaryEvaluator()
    summary = "Fossil fuels drive climate change, warming the Earth by 1.1 degrees Celsius."

    intrinsic_only = ev.evaluate(EN_SOURCE, summary)["overall_quality"]
    assert intrinsic_only["basis"] == "intrinsic"
    assert intrinsic_only["reference_score"] is None
    # Weights of the available components must renormalize to 1.
    assert math.isclose(sum(intrinsic_only["weights"].values()), 1.0, abs_tol=1e-3)

    with_ref = ev.evaluate(
        EN_SOURCE, summary, reference_summary="The Earth has warmed 1.1 degrees."
    )["overall_quality"]
    assert with_ref["basis"] == "reference+intrinsic"
    assert with_ref["reference_score"] is not None
    # The final score is the mean of the two family scores.
    assert math.isclose(
        with_ref["score"],
        0.5 * with_ref["reference_score"] + 0.5 * with_ref["intrinsic_score"],
        abs_tol=5e-4,
    )


def test_ranking_omits_rouge_without_reference():
    ev = SummaryEvaluator()
    comparison = ev.compare_summaries(
        EN_SOURCE,
        {
            "extractive": "Burning fossil fuels generates greenhouse gas emissions.",
            "abstractive": "Fossil fuel use warms the planet.",
        },
    )
    assert "by_rouge_l" not in comparison["rankings"]
    assert "by_factual_consistency" in comparison["rankings"]
    assert len(comparison["rankings"]["by_overall_quality"]) == 2

    with_ref = ev.compare_summaries(
        EN_SOURCE,
        {"a": "Fossil fuels warm the Earth.", "b": "Renewables are cheaper."},
        reference_summary="Fossil fuels warm the Earth.",
    )
    assert with_ref["rankings"]["by_rouge_l"][0][0] == "a"


# -------------------------------------------------------------- statistics

def test_bootstrap_ci_brackets_the_mean():
    stats = bootstrap_mean_ci([0.40, 0.45, 0.50, 0.55, 0.60])
    assert stats["mean"] == 0.5
    assert stats["ci_low"] < 0.5 < stats["ci_high"]
    assert stats["n"] == 5


def test_bootstrap_is_deterministic():
    values = [0.1, 0.3, 0.42, 0.55]
    assert bootstrap_mean_ci(values) == bootstrap_mean_ci(values)


def test_paired_bootstrap_detects_consistent_difference():
    a = [0.50, 0.52, 0.55, 0.51, 0.53, 0.54]
    b = [0.40, 0.41, 0.44, 0.39, 0.42, 0.43]
    result = paired_bootstrap_test(a, b)
    assert result["mean_difference"] > 0.1
    assert result["p_value"] < 0.05

    # Identical systems: no detectable difference.
    same = paired_bootstrap_test(a, list(a))
    assert same["mean_difference"] == 0.0
    assert same["p_value"] == 1.0


def test_too_few_documents_yields_no_p_value():
    """n=2 would otherwise produce a spuriously tiny p from 3 distinct resamples."""
    for a, b in ([[0.7], [0.2]], [[0.7, 0.8], [0.2, 0.3]], [[0.7, 0.8, 0.9], [0.2, 0.3, 0.4]]):
        result = paired_bootstrap_test(a, b)
        assert result["p_value"] is None
        assert "too few" in result["note"].lower()
        assert result["mean_difference"] == 0.5
    # At the threshold, a test is performed.
    enough = paired_bootstrap_test([0.7, 0.8, 0.9, 0.75, 0.85], [0.2, 0.3, 0.4, 0.25, 0.35])
    assert enough["p_value"] is not None


def test_effect_size_and_holm_correction():
    assert cohens_d_paired([0.5, 0.5, 0.5], [0.4, 0.4, 0.4]) is not None
    corrected = holm_bonferroni({"a vs b": 0.01, "a vs c": 0.04, "b vs c": 0.5})
    # Holm: adjusted p is (m - rank) * p, made monotonic.
    # 0.01*3 = 0.03 survives; 0.04*2 = 0.08 does not; 0.5*1 stays 0.5.
    assert corrected["a vs b"]["p_adjusted"] == 0.03
    assert corrected["a vs b"]["significant"] is True
    assert corrected["a vs c"]["p_adjusted"] == 0.08
    assert corrected["a vs c"]["significant"] is False
    assert corrected["b vs c"]["p_adjusted"] == 0.5


# ----------------------------------------------- faithfulness matching rules

def test_possessives_do_not_count_as_unsupported():
    """A summary saying "Einstein's theory" is supported by a source saying "Einstein"."""
    from backend.evaluation.text_utils import normalize_for_match, phrase_in_text

    source = normalize_for_match(
        "Hegerfeldt showed that Einstein locality fails. The Paris Agreement was signed."
    )
    assert phrase_in_text("Einstein's", source)
    assert phrase_in_text("Hegerfeldt's", source)
    # Multi-word names may be reordered; short connectors must not veto a match.
    assert phrase_in_text("Agreement of Paris", source)
    # Genuinely absent names stay flagged — the metric must not become vacuous.
    assert not phrase_in_text("Calabi-Yau", source)
    assert not phrase_in_text("Seiberg-Witten", source)


def test_faithfulness_scores_paraphrase_without_penalising_possessives():
    ev = SummaryEvaluator()
    source = "Hegerfeldt proved the theorem in 1974. Einstein disagreed with the result."
    faithful = ev.evaluate(source, "Hegerfeldt's theorem of 1974 contradicted Einstein's view.")
    assert faithful["factual_consistency"]["score"] == 1.0

    invented = ev.evaluate(source, "Schrodinger's cat appeared in 1999 alongside Calabi-Yau space.")
    assert invented["factual_consistency"]["score"] < 0.5


def test_hyphen_spacing_does_not_count_as_fabrication():
    """
    The arXiv corpus is distributed pre-tokenized as "taylor - couette" while a
    model writes "Taylor-Couette". That is typography, not fabrication.
    """
    from backend.evaluation.text_utils import normalize_for_match, phrase_in_text

    source = normalize_for_match(
        "a rayleigh - stable taylor - couette ( tc ) flow in the potsdam - rossendorf experiment"
    )
    assert phrase_in_text("Taylor-Couette", source)
    assert phrase_in_text("Potsdam-Rossendorf", source)
    assert phrase_in_text("Rayleigh-stable", source)
    # A genuinely different name is still flagged: the metric must not go blind.
    assert not phrase_in_text("Couette-Poiseuille", source)
