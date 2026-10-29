"""
Statistical tools for comparing summarization systems.

A single document cannot support a claim that one model is better than
another: the difference observed on one text is one sample. These functions
aggregate per-document scores across a document set and quantify the
uncertainty, which is what turns a table of numbers into a result.

Implemented:
  - bootstrap_mean_ci      : percentile bootstrap CI for a mean
  - paired_bootstrap_test  : paired bootstrap resampling test (Koehn, 2004)
  - wilcoxon_test          : Wilcoxon signed-rank test (non-parametric, paired)
  - cohens_d_paired        : standardized effect size for paired samples
  - holm_bonferroni        : multiple-comparison correction across model pairs
  - compare_systems        : the whole pipeline over a {system: [scores]} dict
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_RESAMPLES = 10000
DEFAULT_ALPHA = 0.05
# Minimum number of documents before a p-value is reported at all.
#
# A paired bootstrap over n=2 documents draws from only three distinct
# resamples; when both documents happen to favour the same system it returns
# p ≈ 2/(n_resamples+1) — an arbitrarily small number that reflects the
# resampling procedure, not evidence. Below this threshold the difference is
# reported without a p-value, which is the honest result for a two-document
# comparison.
MIN_N_FOR_SIGNIFICANCE = 5
# Fixed seed: the bootstrap is a randomized procedure, and a thesis result
# should be reproducible to the digit.
DEFAULT_SEED = 20260926


def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """
    Percentile bootstrap confidence interval for the mean.

    Args:
        values: Per-document scores for one system.
        n_resamples: Bootstrap replicates.
        alpha: 0.05 gives a 95% interval.

    Returns:
        mean, ci_low, ci_high, std, n.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    n = arr.size
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "std": None, "n": 0}
    if n == 1:
        value = float(arr[0])
        return {
            "mean": round(value, 4),
            "ci_low": round(value, 4),
            "ci_high": round(value, 4),
            "std": 0.0,
            "n": 1,
            "note": "Single observation: no interval can be estimated.",
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
        "std": round(float(arr.std(ddof=1)), 4),
        "n": int(n),
    }


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """
    Paired bootstrap resampling test (Koehn, 2004), two-sided.

    Resamples documents with replacement, keeping each document's pair of
    scores together, and asks how often the sign of the mean difference
    disagrees with the observed sign. Appropriate here because both systems
    are evaluated on exactly the same documents, and it makes no normality
    assumption.

    Returns:
        mean_difference (A - B), p_value, and the bootstrap CI of the difference.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.size != b.size:
        raise ValueError("paired test requires equal-length score lists")
    n = a.size
    if n == 0:
        return {"mean_difference": None, "p_value": None, "n": 0}

    diff = a - b
    observed = float(diff.mean())
    if n < MIN_N_FOR_SIGNIFICANCE:
        return {
            "mean_difference": round(observed, 4),
            "p_value": None,
            "n": int(n),
            "note": (
                f"n={n} document(s): too few for a significance test "
                f"(at least {MIN_N_FOR_SIGNIFICANCE} are required). The "
                f"difference is descriptive only."
            ),
        }

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = diff[idx].mean(axis=1)

    # Two-sided p: how often the resampled difference lands on the other side
    # of zero from the observed difference (ties counted as disagreement).
    if observed >= 0:
        disagreements = int(np.sum(resampled <= 0))
    else:
        disagreements = int(np.sum(resampled >= 0))
    p_value = min(1.0, 2.0 * (disagreements + 1) / (n_resamples + 1))

    low, high = np.percentile(resampled, [2.5, 97.5])
    return {
        "mean_difference": round(observed, 4),
        "p_value": round(float(p_value), 4),
        "diff_ci_low": round(float(low), 4),
        "diff_ci_high": round(float(high), 4),
        "n": int(n),
    }


def wilcoxon_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict:
    """
    Wilcoxon signed-rank test on paired per-document scores.

    Reported alongside the bootstrap as a second, rank-based opinion. It needs
    a handful of non-tied pairs to be meaningful and is skipped otherwise.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    non_tied = int(np.sum(a != b))
    if a.size < 3 or non_tied < 3:
        return {
            "statistic": None,
            "p_value": None,
            "skipped": True,
            "reason": f"needs at least 3 differing pairs, has {non_tied}",
        }
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return {
            "statistic": round(float(result.statistic), 4),
            "p_value": round(float(result.pvalue), 4),
            "skipped": False,
        }
    except Exception as exc:
        return {"statistic": None, "p_value": None, "skipped": True, "reason": str(exc)}


def cohens_d_paired(scores_a: Sequence[float], scores_b: Sequence[float]) -> Optional[float]:
    """Cohen's d for paired samples: mean difference over its standard deviation."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.size < 2:
        return None
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return round(float(diff.mean() / sd), 4)


def interpret_effect_size(d: Optional[float]) -> str:
    """Conventional labels for |d| (Cohen, 1988)."""
    if d is None:
        return "not estimable"
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def holm_bonferroni(
    p_values: Dict[str, Optional[float]], alpha: float = DEFAULT_ALPHA
) -> Dict[str, Dict]:
    """
    Holm-Bonferroni correction over a family of comparisons.

    Comparing k models pairwise means k(k-1)/2 tests, so uncorrected p-values
    overstate significance. Holm is uniformly more powerful than Bonferroni
    and needs no independence assumption.
    """
    valid = {k: v for k, v in p_values.items() if v is not None}
    if not valid:
        return {k: {"p_value": None, "significant": None} for k in p_values}

    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: Dict[str, Dict] = {}
    previous_adjusted = 0.0
    still_rejecting = True
    for rank, (key, p) in enumerate(ordered):
        adjusted = min(1.0, max(previous_adjusted, (m - rank) * p))
        previous_adjusted = adjusted
        if adjusted > alpha:
            still_rejecting = False
        out[key] = {
            "p_value": round(p, 4),
            "p_adjusted": round(adjusted, 4),
            "significant": bool(still_rejecting and adjusted <= alpha),
            "alpha": alpha,
        }
    for key in p_values:
        if key not in out:
            out[key] = {"p_value": None, "p_adjusted": None, "significant": None}
    return out


def compare_systems(
    scores_by_system: Dict[str, List[float]],
    alpha: float = DEFAULT_ALPHA,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Dict:
    """
    Full comparison of several systems scored on the same documents.

    Args:
        scores_by_system: {system name: per-document scores, in the same
            document order for every system}.

    Returns:
        Per-system means with bootstrap CIs, every pairwise test with
        Holm-corrected p-values, and the document count the result rests on.
    """
    systems = [s for s, v in scores_by_system.items() if v]
    summary = {
        s: bootstrap_mean_ci(scores_by_system[s], n_resamples=n_resamples, alpha=alpha)
        for s in systems
    }

    pairwise: Dict[str, Dict] = {}
    raw_p: Dict[str, Optional[float]] = {}
    for i, a in enumerate(systems):
        for b in systems[i + 1 :]:
            va, vb = scores_by_system[a], scores_by_system[b]
            if len(va) != len(vb):
                continue
            key = f"{a} vs {b}"
            boot = paired_bootstrap_test(va, vb, n_resamples=n_resamples)
            d = cohens_d_paired(va, vb)
            pairwise[key] = {
                "system_a": a,
                "system_b": b,
                "mean_difference": boot["mean_difference"],
                "bootstrap_p": boot.get("p_value"),
                "diff_ci_low": boot.get("diff_ci_low"),
                "diff_ci_high": boot.get("diff_ci_high"),
                "wilcoxon": wilcoxon_test(va, vb),
                "cohens_d": d,
                "effect_size": interpret_effect_size(d),
            }
            raw_p[key] = boot.get("p_value")

    corrected = holm_bonferroni(raw_p, alpha=alpha)
    for key, correction in corrected.items():
        if key in pairwise:
            pairwise[key].update(
                {
                    "p_adjusted": correction.get("p_adjusted"),
                    "significant": correction.get("significant"),
                }
            )

    n_docs = max((len(v) for v in scores_by_system.values()), default=0)
    return {
        "per_system": summary,
        "pairwise": pairwise,
        "n_documents": n_docs,
        "alpha": alpha,
        "n_resamples": n_resamples,
        "method": (
            "Paired bootstrap resampling (Koehn, 2004) with percentile 95% CIs, "
            "Wilcoxon signed-rank as a rank-based check, and Holm-Bonferroni "
            "correction across all pairwise comparisons."
        ),
        "significance_tested": n_docs >= MIN_N_FOR_SIGNIFICANCE,
        "min_n_for_significance": MIN_N_FOR_SIGNIFICANCE,
        "caveat": (
            f"Conclusions rest on n={n_docs} documents. With a small document set, "
            "wide intervals and non-significant differences are expected and should "
            "be reported as such."
            if n_docs >= MIN_N_FOR_SIGNIFICANCE
            else (
                f"n={n_docs} documents: differences are reported descriptively, without "
                f"p-values. At least {MIN_N_FOR_SIGNIFICANCE} documents are needed before "
                f"a significance test carries information."
            )
        ),
    }
