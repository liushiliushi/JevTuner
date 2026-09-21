"""Core objective for closed-set probabilistic decisions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def choice_logits(
    vocabulary_logits: torch.Tensor,
    decision_token_ids: torch.Tensor,
    choice_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select one logit per decision slot and mask unavailable choices."""
    if vocabulary_logits.ndim != 2:
        raise ValueError("vocabulary_logits must have shape [batch, vocabulary]")
    if decision_token_ids.ndim != 1:
        raise ValueError("decision_token_ids must have shape [max_choices]")

    selected = vocabulary_logits.index_select(1, decision_token_ids)
    if choice_mask is not None:
        if choice_mask.shape != selected.shape:
            raise ValueError("choice_mask must match the selected choice logits")
        if not torch.all(choice_mask.any(dim=1)):
            raise ValueError("every example must expose at least one choice")
        selected = selected.masked_fill(~choice_mask.bool(), float("-inf"))
    return selected


def decision_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute Tokenized Multi-class Brier Loss.

    Unlike ConfTuner's scalar confidence objective, every candidate decision is
    a class and the target is a one-hot distribution over those classes.
    """
    probabilities = decision_probabilities(logits)
    targets = F.one_hot(labels.long(), num_classes=logits.shape[-1]).to(
        probabilities.dtype
    )
    return ((probabilities - targets) ** 2).sum(dim=-1).mean()


def decision_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return a normalized probability distribution over available choices."""
    return F.softmax(logits.float(), dim=-1)
