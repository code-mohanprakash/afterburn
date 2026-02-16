"""Shared NLI (Natural Language Inference) model for semantic analysis.

Uses cross-encoder/nli-deberta-v3-small for:
- Sycophancy detection: does the response entail a false claim?
- Answer verification: does the response contain the correct answer?
- Strategy classification: zero-shot NLI-based text classification

The model is loaded lazily on first use. If `transformers` is not installed,
all functions gracefully return None to fall back to regex-based methods.

Labels: 0=contradiction, 1=entailment, 2=neutral
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

_nli_model = None
_nli_tokenizer = None
_nli_available: bool | None = None


@dataclass(frozen=True)
class NLIResult:
    """Result of NLI inference on a single premise-hypothesis pair."""

    contradiction: float
    entailment: float
    neutral: float


def is_nli_available() -> bool:
    """Check if NLI model can be loaded (transformers + torch installed)."""
    global _nli_available
    if _nli_available is not None:
        return _nli_available
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        _nli_available = True
    except ImportError:
        _nli_available = False
        logger.debug("transformers or torch not available, NLI features disabled")
    return _nli_available


_nli_load_failed = False


def _get_nli_model() -> tuple[object | None, object | None]:
    """Lazy singleton for NLI model and tokenizer.

    Returns (None, None) if model can't be loaded (missing packages,
    disk space, network issues, etc.).
    """
    global _nli_model, _nli_tokenizer, _nli_load_failed
    if _nli_load_failed:
        return None, None
    if _nli_model is None:
        if not is_nli_available():
            return None, None
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("Loading NLI model: %s", NLI_MODEL_NAME)
            _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)  # type: ignore[no-untyped-call]
            _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
            _nli_model.eval()
        except OSError as e:
            logger.warning("NLI load failed (file/network): %s", e)
            _nli_load_failed = True
            return None, None
        except ImportError as e:
            logger.warning("NLI load failed (missing deps): %s", e)
            _nli_load_failed = True
            return None, None
        except RuntimeError as e:
            logger.warning("NLI load failed (torch/model): %s", e)
            _nli_load_failed = True
            return None, None
    return _nli_model, _nli_tokenizer


def nli_predict(premise: str, hypothesis: str) -> NLIResult | None:
    """Run NLI inference on a single premise-hypothesis pair.

    Returns NLIResult with contradiction/entailment/neutral probabilities,
    or None if NLI is not available.
    """
    model, tokenizer = _get_nli_model()
    if model is None or tokenizer is None:
        return None

    import torch

    features = tokenizer(  # type: ignore[operator]
        [premise],
        [hypothesis],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**features).logits  # type: ignore[operator]  # [1, 3]
        probs = torch.softmax(logits, dim=-1)[0]

    return NLIResult(
        contradiction=probs[0].item(),
        entailment=probs[1].item(),
        neutral=probs[2].item(),
    )


def nli_predict_batch(
    premises: list[str], hypotheses: list[str], batch_size: int = 16
) -> list[NLIResult] | None:
    """Run NLI inference on multiple premise-hypothesis pairs.

    Returns list of NLIResult, or None if NLI is not available.
    """
    model, tokenizer = _get_nli_model()
    if model is None or tokenizer is None:
        return None

    import torch

    results: list[NLIResult] = []

    for i in range(0, len(premises), batch_size):
        batch_premises = premises[i : i + batch_size]
        batch_hypotheses = hypotheses[i : i + batch_size]

        features = tokenizer(  # type: ignore[operator]
            batch_premises,
            batch_hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**features).logits  # type: ignore[operator]
            probs = torch.softmax(logits, dim=-1)

        for j in range(len(batch_premises)):
            results.append(
                NLIResult(
                    contradiction=probs[j][0].item(),
                    entailment=probs[j][1].item(),
                    neutral=probs[j][2].item(),
                )
            )

    return results


def zero_shot_classify(
    text: str,
    candidate_labels: list[str],
    hypothesis_template: str = "This example is {}.",
) -> dict[str, float] | None:
    """Zero-shot text classification using NLI.

    For each label, constructs hypothesis from template and runs NLI.
    Returns dict mapping label → probability (softmax across all labels'
    entailment logits), or None if NLI is not available.

    This mirrors the HuggingFace zero-shot-classification pipeline.
    """
    model, tokenizer = _get_nli_model()
    if model is None or tokenizer is None:
        return None

    import torch

    hypotheses = [hypothesis_template.format(label) for label in candidate_labels]
    premises = [text] * len(candidate_labels)

    features = tokenizer(  # type: ignore[operator]
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**features).logits  # type: ignore[operator]  # [num_labels, 3]
        # Extract entailment logits (index 1) for each label
        entail_logits = logits[:, 1]  # [num_labels]
        # Softmax across labels (competition)
        scores = torch.softmax(entail_logits, dim=0)

    return {label: scores[i].item() for i, label in enumerate(candidate_labels)}
