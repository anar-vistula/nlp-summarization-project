"""
Canonical registry of every summarization system in the study.

The interactive UI only offers models that suit the detected language. The
experiment pipeline deliberately does NOT: it runs every model over every
document, including the mismatched combinations (an English-only BART on a
Turkish text, a Turkish mBART on an English one). Those cells are the point —
they measure what happens when a model meets a language it was not trained
for, which is otherwise only assumed.

Each entry therefore records what the model was trained for, and the runner
labels every result as in-language or cross-language so the two can be
analysed separately.
"""

import os
from typing import Dict, List, Optional

# trained_for: the language(s) the model was actually trained on.
#   "en"    English only
#   "tr"    Turkish only
#   "multi" explicitly multilingual
#   "any"   language-agnostic algorithm (no training data at all)
ALL_MODELS: List[Dict] = [
    {
        "key": "extractive",
        "name": "TextRank (Extractive)",
        "type": "extractive",
        "trained_for": "any",
        "note": "Unsupervised graph ranking; language support comes from stopwords and the sentence splitter.",
    },
    {
        "key": "distilbart",
        "name": "DistilBART-CNN",
        "type": "hf",
        "trained_for": "en",
        "note": "Distilled BART fine-tuned on CNN/DailyMail. 1024-token input.",
    },
    {
        "key": "bart-large",
        "name": "BART-Large-CNN",
        "type": "hf",
        "trained_for": "en",
        "note": "Full BART-Large fine-tuned on CNN/DailyMail. 1024-token input.",
    },
    {
        "key": "pegasus",
        "name": "Pegasus-CNN/DailyMail",
        "type": "hf",
        "trained_for": "en",
        "note": "Gap-sentence pre-training aimed at summarization.",
    },
    {
        "key": "mt5-xlsum",
        "name": "mT5-XLSum (Multilingual)",
        "type": "hf",
        "trained_for": "multi",
        "note": "mT5 fine-tuned on XLSum in 44 languages. 512-token input.",
    },
    {
        "key": "mt5-turkish-finetuned",
        "name": "mT5-Turkish (Fine-tuned)",
        "type": "hf",
        "trained_for": "tr",
        "note": "Locally fine-tuned on XLSum-Turkish; only present if models/ holds it.",
    },
    {
        "key": "led-large-book",
        "name": "LED-Large-BookSum (GPU)",
        "type": "colab",
        "trained_for": "en",
        "note": "Longformer encoder-decoder, 16k-token input, fine-tuned on BookSum.",
    },
    {
        "key": "flan-t5-large",
        "name": "Flan-T5-Large (GPU)",
        "type": "colab",
        "trained_for": "en",
        "note": "Instruction-tuned T5; summarization via prompt.",
    },
    {
        "key": "turkish-mbart",
        "name": "mBART Turkish (GPU)",
        "type": "colab",
        "trained_for": "tr",
        "note": "mBART-Large fine-tuned on Turkish MLSum.",
    },
    {
        "key": "qwen-llm",
        "name": "Qwen2.5-7B-Instruct (GPU)",
        "type": "colab",
        "trained_for": "multi",
        "note": "Open instruction-tuned LLM in 4-bit on the Colab T4; free stand-in for a paid LLM.",
    },
    {
        "key": "gpt-4",
        "name": "GPT-4 (OpenAI)",
        "type": "paid",
        "trained_for": "multi",
        "note": "Legacy 8k context: long documents overflow, since paid models are not chunked.",
    },
    {
        "key": "gpt-4o-mini",
        "name": "GPT-4o Mini (OpenAI)",
        "type": "paid",
        "trained_for": "multi",
        "note": "128k context; the whole document is sent in one request.",
    },
    {
        "key": "claude-sonnet",
        "name": "Claude Sonnet (Anthropic)",
        "type": "paid",
        "trained_for": "multi",
        "note": "Needs a working ANTHROPIC_API_KEY.",
    },
    {
        "key": "claude-haiku",
        "name": "Claude Haiku (Anthropic)",
        "type": "paid",
        "trained_for": "multi",
        "note": "Needs a working ANTHROPIC_API_KEY.",
    },
]

BY_KEY: Dict[str, Dict] = {m["key"]: m for m in ALL_MODELS}


def language_match(model_key: str, doc_language: str) -> str:
    """
    How a model relates to the language of the document it is given.

    Returns "in_language" (trained on it, or language-agnostic),
    "multilingual" (trained on many languages including, presumably, this one),
    or "cross_language" (trained on a different language entirely).
    """
    model = BY_KEY.get(model_key)
    if not model:
        return "unknown"
    trained = model["trained_for"]
    if trained == "any":
        return "in_language"
    if trained == "multi":
        return "multilingual"
    return "in_language" if trained == doc_language else "cross_language"


def availability(check_colab: bool = True) -> Dict[str, Dict]:
    """
    Which models can actually run right now, and why not when they cannot.

    Checked here rather than at run time so an experiment fails fast with a
    clear list instead of producing 100 identical errors.
    """
    from backend.summarizers.abstractive import finetuned_turkish_available
    from backend.summarizers.paid_models import ColabGPUClient, PaidModelSummarizer

    paid_available = set(PaidModelSummarizer().get_available_models())
    colab_models: List[str] = []
    if check_colab:
        client = ColabGPUClient()
        if client.is_available():
            colab_models = client.get_available_models()

    status: Dict[str, Dict] = {}
    for model in ALL_MODELS:
        key, model_type = model["key"], model["type"]
        if model_type in ("extractive", "hf"):
            if key == "mt5-turkish-finetuned" and not finetuned_turkish_available():
                status[key] = {"available": False, "reason": "models/mt5-turkish-finetuned not present"}
            else:
                status[key] = {"available": True, "reason": ""}
        elif model_type == "colab":
            if not colab_models:
                status[key] = {"available": False, "reason": "Colab GPU server not reachable"}
            elif key not in colab_models:
                status[key] = {"available": False, "reason": f"Colab server does not host '{key}'"}
            else:
                status[key] = {"available": True, "reason": ""}
        elif model_type == "paid":
            if key in paid_available:
                status[key] = {"available": True, "reason": ""}
            else:
                status[key] = {"available": False, "reason": "API key missing or placeholder"}
    return status


def resolve(
    requested: Optional[List[str]] = None,
    include_paid: bool = False,
    include_colab: bool = True,
    check_colab: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Work out the model set for an experiment.

    Args:
        requested: explicit model keys; None means "everything allowed by the
            include_* flags".
        include_paid: paid APIs cost real money per document, so they are
            opt-in even when a key is configured.
        include_colab: Colab models need the notebook to be running.

    Returns:
        {"models": [...runnable...], "skipped": [...with reasons...]}
    """
    status = availability(check_colab=check_colab)
    keys = requested if requested is not None else [m["key"] for m in ALL_MODELS]

    models, skipped = [], []
    for key in keys:
        model = BY_KEY.get(key)
        if not model:
            skipped.append({"key": key, "reason": "unknown model key"})
            continue
        if model["type"] == "paid" and not include_paid:
            skipped.append({"key": key, "name": model["name"], "reason": "paid models not enabled (--paid)"})
            continue
        if model["type"] == "colab" and not include_colab:
            skipped.append({"key": key, "name": model["name"], "reason": "Colab models disabled"})
            continue
        state = status.get(key, {"available": False, "reason": "unknown"})
        if not state["available"]:
            skipped.append({"key": key, "name": model["name"], "reason": state["reason"]})
            continue
        models.append(dict(model))
    return {"models": models, "skipped": skipped}
