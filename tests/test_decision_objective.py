import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from llama_recipes.utils.decision_objective import (  # noqa: E402
    choice_logits,
    decision_loss,
    decision_probabilities,
)


def test_choice_logits_selects_and_masks_slots():
    vocabulary_logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    selected = choice_logits(
        vocabulary_logits,
        torch.tensor([4, 2, 1]),
        torch.tensor([[True, True, False]]),
    )
    assert selected[0, :2].tolist() == [4.0, 2.0]
    assert torch.isneginf(selected[0, 2])


def test_probabilities_are_normalized_over_available_choices():
    logits = torch.tensor([[2.0, 1.0, float("-inf")]])
    probabilities = decision_probabilities(logits)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
    assert probabilities[0, 2] == 0


def test_brier_loss_rewards_more_probability_on_correct_decision():
    label = torch.tensor([0])
    good = decision_loss(torch.tensor([[4.0, 0.0]]), label)
    bad = decision_loss(torch.tensor([[0.0, 4.0]]), label)
    assert good < bad


def test_brier_loss_matches_multiclass_definition():
    logits = torch.log(torch.tensor([[0.2, 0.3, 0.5]]))
    loss = decision_loss(logits, torch.tensor([2]))
    expected = 0.2**2 + 0.3**2 + (0.5 - 1.0) ** 2
    assert torch.allclose(loss, torch.tensor(expected))
