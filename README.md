# The project

Our project for Intro to LLMs. We aim to find the gap between prompting and finetuning (LoRA) of Gemma 4 E4B for multi label classification of emotions from subtitles, and use Qwen3.5 4B as a secondary model for just prompting.

We make use of the XED dataset, English and Dutch subtitles.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/joshvanvliet/RUG-llm-course.git
uv sync
```

The XED data comes as a git submodule. Gemma 4 is on Hugging Face, but you have to log in with `hf auth login` first.

## Data

We use `Data/dataset_maker.py` to match English and Dutch lines in the XED alignment file, and to split the 1,429 pairs 70/20/10 (multi-label stratified, seed 0).
Duplicates/parallel lines always end up in the same part of the split.
The split is saved in `splits/nl.tsv`, so every run uses the same one.

## Reproducing our results

```bash
uv run python -m Training.train 0            # English, Dutch and bilingual LoRA, seed 0
uv run python -m Training.train 1            # same with seed 1
uv run python -m Evaluation.evaluate develop # picks the yes-threshold per model
uv run python -m Evaluation.evaluate test    # final scores, mean ± sd over both seeds
uv run python -m Prompting.prompting gemma test
uv run python -m Prompting.prompting qwen test
uv run python -m Prompting.prompting gemma develop alternative  # prompt wording check
```

The results end up in `results/`, the raw model outputs and probabilities in `logs/`. Model weights are not committed.

## Settings

- LoRA: rank 16, alpha 16, dropout 0.05 on all attention and MLP projections, learning rate 2e-4,
  batch 1 with 8 gradient accumulation steps, 2 epochs to keep the lowest develop loss, max length 256, bf16,
  no quantization, seeds 0 and 1.
- The finetuned model answers yes or no per emotion. An emotion counts when its probability for yes is above a
  threshold we choose on the develop set.
- Prompting: greedy decoding, with max. 64 new tokens, thinking off. Five shot uses five fixed training pairs
  (seed 0), that are shown in the language that is evaluated. Outputs that are not a JSON list count as malformed (resulting in no emotion).
- Metrics: scikit-learn, zero division set to 0, fixed label order.
- Our hardware: one RTX 3090 (24 GB VRAM).
