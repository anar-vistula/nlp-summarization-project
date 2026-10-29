"""
Tokenization helpers shared by all metrics.

The default tokenizer of the `rouge_score` package strips every non-ASCII
character, which silently destroys Turkish text ("İstanbul'un" -> "stanbul un",
"öğretmen" -> "retmen"). Every metric in this package therefore uses the
Unicode-aware tokenizer defined here, so English and Turkish are measured on
the same footing.
"""

import re
import unicodedata
from typing import List, Optional, Sequence, Set, Tuple

# Word = one or more Unicode letters/digits, keeping intra-word apostrophes
# and hyphens (so "COVID-19" and "Türkiye'nin" stay single tokens).
_WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)

# Numbers, including decimals, thousands separators and percentages.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*\s?%?", re.UNICODE)

# Scripts that do not separate words with spaces. Chinese, Japanese and the
# CJK compatibility range are counted one character per word, because a
# whitespace split returns 1-10 "words" for a full paragraph of Chinese and so
# corrupts every length-derived measure (compression ratio, percentage of a
# word target, and length used as a covariate). This is not hypothetical: the
# 7B model answered 12 of 100 Turkish documents in Chinese.
_UNSPACED_SCRIPT_RE = re.compile(
    "["
    "\u3040-\u30ff"      # Hiragana, Katakana
    "\u3400-\u4dbf"      # CJK Extension A
    "\u4e00-\u9fff"      # CJK Unified Ideographs
    "\uf900-\ufaff"      # CJK Compatibility Ideographs
    "]"
)


def count_words(text: str) -> int:
    """
    Word count that survives a change of script.

    For any text containing no unspaced-script characters this returns exactly
    `len(text.split())`, so it cannot perturb previously computed results for
    English or Turkish. Where such characters appear, each counts as one word
    and the remaining text is still split on whitespace.
    """
    if not text:
        return 0
    unspaced = len(_UNSPACED_SCRIPT_RE.findall(text))
    if not unspaced:
        return len(text.split())
    return unspaced + len(_UNSPACED_SCRIPT_RE.sub(" ", text).split())


# Languages for which the Porter stemmer (English-only) is meaningful.
STEMMABLE_LANGS = {"en"}


def _lower(text: str, lang: str) -> str:
    """
    Lowercase with the Turkish dotted-I handled.

    Python maps "İ" to "i" + U+0307 (combining dot above), so "İstanbul"
    becomes a two-codepoint "i̇stanbul" that matches nothing. Turkish text is
    pre-mapped, and the stray combining dot is stripped for every language as a
    safety net.
    """
    if lang == "tr":
        text = text.replace("İ", "i").replace("I", "ı")
    return text.lower().replace("̇", "")


def tokenize(text: str, lang: str = "en", stem: bool = False) -> List[str]:
    """
    Split text into lowercase word tokens, preserving non-ASCII letters.

    Args:
        text: Input text.
        lang: Language code. Stemming is only applied to English.
        stem: Apply Porter stemming (English only, matches ROUGE convention).
    """
    if not text:
        return []
    tokens = [_lower(m.group(0), lang) for m in _WORD_RE.finditer(text)]
    if stem and lang in STEMMABLE_LANGS:
        from nltk.stem.porter import PorterStemmer

        stemmer = PorterStemmer()
        # ROUGE only stems tokens longer than 3 characters.
        tokens = [stemmer.stem(t) if len(t) > 3 else t for t in tokens]
    return tokens


class UnicodeRougeTokenizer:
    """
    Tokenizer object accepted by `rouge_score.rouge_scorer.RougeScorer`.
    Keeps Turkish (and any other non-ASCII) characters intact.
    """

    def __init__(self, lang: str = "en", stem: bool = True):
        self.lang = lang
        self.stem = stem

    def tokenize(self, text: str) -> List[str]:
        return tokenize(text, lang=self.lang, stem=self.stem)


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """All n-grams of a token sequence, in order."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_set(tokens: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
    """Unique n-grams of a token sequence."""
    return set(ngrams(tokens, n))


def extract_numbers(text: str) -> List[str]:
    """Numeric mentions in the text, normalized to a comparable form."""
    out = []
    for raw in _NUMBER_RE.findall(text or ""):
        cleaned = raw.strip().rstrip("%").strip()
        cleaned = cleaned.replace(",", "")
        cleaned = cleaned.rstrip(".")
        if cleaned:
            out.append(cleaned)
    return out


def extract_capitalized_phrases(text: str, lang: str = "en") -> List[str]:
    """
    Proper-noun candidates: capitalized words that are not sentence-initial,
    merged into phrases ("Paris Agreement", "Mustafa Kemal").

    This is a lightweight, dependency-free stand-in for a named entity
    recognizer. It over-generates on German-style capitalization but is
    adequate for English and Turkish, and it is only ever used to compare a
    summary against its own source text, so systematic errors cancel out.
    """
    phrases: List[str] = []
    for sentence in re.split(r"(?<=[.!?…])\s+|\n+", text or ""):
        words = sentence.split()
        current: List[str] = []
        for idx, word in enumerate(words):
            core = word.strip("\"'“”‘’()[]{}.,;:!?…–—")
            is_cap = bool(core) and core[:1].isupper() and not core.isupper()
            # Skip the first word of a sentence: capitalization is grammatical.
            if is_cap and idx > 0 and len(core) > 1:
                current.append(core)
            else:
                if current:
                    phrases.append(" ".join(current))
                    current = []
        if current:
            phrases.append(" ".join(current))
    return phrases


# Hyphens and dashes vary between a summary and its source for reasons that
# have nothing to do with faithfulness: the arXiv corpus is distributed
# pre-tokenized with spaces around hyphens ("taylor - couette"), while a model
# writes "Taylor-Couette". Left unhandled, every hyphenated technical term in a
# well-formed summary is reported as fabricated.
_HYPHEN_RE = re.compile(r"\s*[-\u2010-\u2015\u2212]\s*")


def normalize_for_match(text: str) -> str:
    """
    Casefold, strip accents, and normalize hyphenation for lenient comparison.

    Hyphen spacing is collapsed on BOTH sides, so "taylor - couette",
    "Taylor-Couette" and "taylor couette" all compare equal. This is a
    typographic difference, not a factual one.
    """
    lowered = (text or "").casefold()
    decomposed = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _HYPHEN_RE.sub(" ", stripped)


def match_token_set(tokens: Sequence[str]) -> Set[str]:
    """
    Token set for term-presence checks, extended with the parts of hyphenated
    and apostrophed tokens.

    Needed because scikit-learn's TF-IDF tokenizer splits "long-term" into
    "long" and "term" while this module keeps it as one token; without the
    split parts, terms extracted by TF-IDF would be reported as missing from a
    summary that plainly contains them.
    """
    out: Set[str] = set()
    for token in tokens:
        out.add(token)
        if "-" in token or "'" in token or "’" in token:
            for part in re.split(r"['’\-]", token):
                if part:
                    out.add(part)
    return out


def term_in_tokens(term: str, tokens: Set[str], min_prefix_len: int = 5) -> bool:
    """
    Check whether a source term is present in a set of summary tokens.

    Exact token match, or prefix match for longer terms so that inflected
    forms count ("climate" ~ "climates", "hükümet" ~ "hükümetin"). Turkish is
    agglutinative, so requiring exact matches would systematically
    under-report term retention for Turkish summaries.

    Substring matching over the raw string is deliberately avoided: it reports
    "ai" as present inside "said".
    """
    parts = term.split()
    if len(parts) > 1:
        # Multi-word term: every part must be present.
        return all(term_in_tokens(p, tokens, min_prefix_len) for p in parts)
    if term in tokens:
        return True
    if len(term) >= min_prefix_len:
        return any(t.startswith(term) for t in tokens)
    return False


# Possessive endings that must not defeat a match: a summary writing
# "Einstein's theory" is supported by a source that says "Einstein".
_POSSESSIVE_RE = re.compile(r"['\u2019]s\b|['\u2019]\B", re.UNICODE)


def strip_possessive(phrase: str) -> str:
    """Remove English possessive endings from each word of a phrase."""
    return _POSSESSIVE_RE.sub("", phrase).strip()


def phrase_in_text(phrase: str, text_normalized: str) -> bool:
    """
    Whether a phrase occurs in already-normalized text.

    Matching is lenient in two documented ways, because the alternative is to
    report ordinary English as unfaithful:

    1. Possessives are stripped ("Einstein's" is supported by "Einstein").
    2. A multi-word proper name counts as supported when every one of its parts
       appears in the source, since summaries routinely reorder or re-join
       names ("the Paris Agreement" vs "Agreement of Paris").

    Still NOT matched, deliberately: a name the source never contains in any
    form, and an expansion the source never spells out ("Friedmann-Robertson-
    Walker" where the source only ever writes "Friedmann"). Those remain
    flagged, and the limitation is stated in the methodology.
    """
    normalized = normalize_for_match(phrase)
    if normalized in text_normalized:
        return True
    bare = normalize_for_match(strip_possessive(phrase))
    if bare and bare in text_normalized:
        return True
    # Ignore short connectors ("of", "de", "la"): requiring them to appear
    # would let a function word veto an otherwise supported name.
    parts = [p for p in re.split(r"[\s]+", bare) if len(p) > 2]
    if len(parts) > 1:
        return all(p in text_normalized for p in parts)
    return False


def sentence_split(text: str, lang: str = "en") -> List[str]:
    """
    Language-aware sentence splitting via NLTK punkt, with a regex fallback.
    Kept here so every metric splits sentences identically.
    """
    import nltk

    from backend.preprocessing.text_preprocessor import PUNKT_LANG_MAP

    punkt_lang = PUNKT_LANG_MAP.get(lang, "english")
    try:
        sentences = nltk.sent_tokenize(text, language=punkt_lang)
    except LookupError:
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            sentences = re.split(r"(?<=[.!?…])\s+", text)
    return [s.strip() for s in sentences if s and s.strip()]


def count_syllables_en(word: str) -> int:
    """Heuristic English syllable count (for Flesch Reading Ease)."""
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    previous_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    if word.endswith("e") and count > 1 and not word.endswith(("le", "ee")):
        count -= 1
    return max(1, count)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns `default` instead of raising on a zero divisor."""
    return numerator / denominator if denominator else default
