"""Train and query a closed-set, probabilistic decision model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_recipes.utils.decision_objective import (
    choice_logits,
    decision_loss,
    decision_probabilities,
)


SYSTEM_PROMPT = (
    "Make the requested decision from the supplied state. The answer must be "
    "exactly one of the numbered choices."
)


def decision_tokens(max_choices: int) -> list[str]:
    if max_choices < 2:
        raise ValueError("max_choices must be at least 2")
    return [f"<|decision_{index}|>" for index in range(max_choices)]


def format_prompt(state: Any, question: str, choices: list[str]) -> str:
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    rendered_choices = "\n".join(
        f"{index}. {choice}" for index, choice in enumerate(choices)
    )
    return (
        f"{SYSTEM_PROMPT}\n\nState:\n{state_text}\n\n"
        f"Question:\n{question}\n\nChoices:\n{rendered_choices}\n\nDecision:"
    )


def validate_example(example: dict[str, Any], max_choices: int) -> None:
    required = {"state", "question", "choices", "label"}
    missing = required - example.keys()
    if missing:
        raise ValueError(f"example is missing fields: {sorted(missing)}")
    choices = example["choices"]
    if not isinstance(choices, list) or not 2 <= len(choices) <= max_choices:
        raise ValueError(f"choices must contain 2..{max_choices} values")
    if not all(isinstance(x, str) for x in choices) or len(set(choices)) != len(choices):
        raise ValueError("choices must be unique strings")
    if not isinstance(example["label"], int) or not 0 <= example["label"] < len(choices):
        raise ValueError("label must be the zero-based index of a choice")


class DecisionDataset(Dataset):
    def __init__(self, path: str, max_choices: int):
        self.examples = []
        with open(path, encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                    validate_example(example, max_choices)
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"invalid example at {path}:{line_number}: {error}") from error
                self.examples.append(example)
        if not self.examples:
            raise ValueError(f"no examples found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class DecisionCollator:
    def __init__(self, tokenizer, max_choices: int, max_length: int):
        self.tokenizer = tokenizer
        self.max_choices = max_choices
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts = [
            format_prompt(item["state"], item["question"], item["choices"])
            for item in examples
        ]
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        choice_mask = torch.zeros(len(examples), self.max_choices, dtype=torch.bool)
        for row, item in enumerate(examples):
            choice_mask[row, : len(item["choices"])] = True
        encoded["choice_mask"] = choice_mask
        encoded["decision_labels"] = torch.tensor(
            [item["label"] for item in examples], dtype=torch.long
        )
        return encoded


def prepare_tokenizer(model_name: str, max_choices: int, padding_side: str = "left"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": decision_tokens(max_choices)}
    )
    return tokenizer


def get_decision_token_ids(tokenizer, max_choices: int, device=None) -> torch.Tensor:
    ids = tokenizer.convert_tokens_to_ids(decision_tokens(max_choices))
    if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in ids:
        raise ValueError("model tokenizer does not contain JevTuner decision tokens")
    return torch.tensor(ids, dtype=torch.long, device=device)


def train(
    model_name: str,
    train_file: str,
    output_dir: str,
    max_choices: int = 32,
    max_length: int = 1024,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    use_peft: bool = True,
    lora_rank: int = 8,
    seed: int = 42,
) -> None:
    """Fine-tune with Tokenized Multi-class Brier Loss."""
    torch.manual_seed(seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    tokenizer = prepare_tokenizer(model_name, max_choices)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))

    if use_peft:
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                modules_to_save=["embed_tokens", "lm_head"],
            ),
        )

    dataset = DecisionDataset(train_file, max_choices)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DecisionCollator(tokenizer, max_choices, max_length),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    token_ids = get_decision_token_ids(tokenizer, max_choices, accelerator.device)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            choice_mask = batch.pop("choice_mask")
            labels = batch.pop("decision_labels")
            with accelerator.accumulate(model):
                output = model(**batch)
                final_logits = output.logits[:, -1, :]
                logits = choice_logits(final_logits, token_ids, choice_mask)
                loss = decision_loss(logits, labels)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += accelerator.gather(loss.detach()).mean().item()
        accelerator.print(
            f"epoch={epoch + 1} decision_brier={running_loss / len(dataloader):.6f}"
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        if use_peft:
            unwrapped = unwrapped.merge_and_unload()
        unwrapped.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)


@torch.inference_mode()
def decide(
    model_name: str,
    state: str,
    question: str,
    choices: str | list[str],
    max_choices: int = 32,
    max_length: int = 1024,
) -> dict[str, Any]:
    """Return one decision and the probability assigned to every choice."""
    if isinstance(choices, str):
        choices = json.loads(choices)
    validate_example(
        {"state": state, "question": question, "choices": choices, "label": 0},
        max_choices,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    encoded = tokenizer(
        format_prompt(state, question, choices),
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)
    output = model(**encoded)
    token_ids = get_decision_token_ids(tokenizer, max_choices, model.device)
    mask = torch.zeros(1, max_choices, dtype=torch.bool, device=model.device)
    mask[:, : len(choices)] = True
    logits = choice_logits(output.logits[:, -1, :], token_ids, mask)
    probabilities = decision_probabilities(logits)[0].cpu().tolist()
    probabilities = probabilities[: len(choices)]
    selected = max(range(len(choices)), key=probabilities.__getitem__)
    result = {
        "decision": choices[selected],
        "probabilities": dict(zip(choices, probabilities)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    fire.Fire({"train": train, "decide": decide})
