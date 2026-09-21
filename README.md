# JevTuner

JevTuner is a small, open implementation of the **decision-first** idea exposed by
TypeSafe AI's Jev API. It builds on the core idea of
[ConfTuner](https://github.com/liushiliushi/ConfTuner): directly training the
probability distribution represented by token logits. Given a state, a typed
question, and a closed set of choices, JevTuner returns a normalized probability
for every choice in one forward pass and optimizes the complete distribution with
Tokenized Multi-class Brier Loss.

> [!IMPORTANT]
> Jev is closed source. TypeSafe publicly names its training method
> "Reinforcement Learning for Calibrated Decisions" (RLCD), but does not publish
> its loss or architecture. This repository is therefore a Jev-style tuner, not
> a reproduction or implementation of TypeSafe's private RLCD algorithm.
> See TypeSafe's [public Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

## Install

```bash
pip install -r requirements.txt
```

## Data format

Use JSON Lines with one decision per line:

```json
{"state":"The customer says they were charged twice.","question":"Which team should handle this?","choices":["billing","technical","sales","other"],"label":0}
```

`label` is the zero-based index of the correct choice. A dataset may mix
different choice strings and different numbers of choices, up to
`--max_choices` (default: 32).

## Train

```bash
cd src
python jevtuner.py train \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --train_file ../examples/decisions.jsonl \
  --output_dir ../checkpoints/jevtuner \
  --use_peft True
```

The model learns reserved decision tokens (`<|decision_0|>`,
`<|decision_1|>`, ...). They are internal class slots: the returned values are
mapped back to the choice strings supplied for each request. This keeps the
output space closed and makes every choice score available from the same final
hidden state.

For predicted choice probabilities `p` and a one-hot target `y`, the objective is:

```text
L = sum_k (p_k - y_k)^2
```

## Decide

```bash
cd src
python jevtuner.py decide \
  --model_name ../checkpoints/jevtuner \
  --state "The customer says they were charged twice." \
  --question "Which team should handle this?" \
  --choices '["billing", "technical", "sales", "other"]'
```

Example output:

```json
{
  "decision": "billing",
  "probabilities": {
    "billing": 0.84,
    "technical": 0.09,
    "sales": 0.03,
    "other": 0.04
  }
}
```

The `decision` is `argmax(probabilities)`; there is no separately generated
confidence string.

## What stayed unchanged

The original ConfTuner/Llama Recipes files remain under `src/llama_recipes` so
the upstream data and model-loading utilities are still available. JevTuner's
new default entry point is `src/jevtuner.py`; the legacy confidence scripts are
kept only for comparison and are not used by it.

## Attribution

This project is derived from ConfTuner and retains its upstream license notices.
"Jev" is a product of TypeSafe AI; this project is unofficial and unaffiliated.
