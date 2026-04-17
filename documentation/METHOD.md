# Methodology for Commit Annotation Using Large Language Models

In this document, the methodology used to evaluate the ability of Large Language Models (LLMs) to annotate software commits from the Linux Kernel repository will be described. 
The aim will be to link this study with the guidelines proposed by Baltes et al.[2], "Guidelines for Empirical Studies in Software Engineering involving Large Language Models".
  
# Methodology

This document describes what was actually implemented and executed in this repository. It is organized into two main phases:

- **Experimentation phase** — running automated annotations with multiple LLMs, controlling context modes, and persisting results.
- **Analysis phase** — evaluating agreement between LLMs and human annotators using reproducible metrics.

## Executive summary

A set of Linux Kernel commits was annotated using a multi-dimensional prompt template (BFC, BPC, PRC, NFC). Results were produced per model and per "round" (context configuration). Execution is orchestrated by `annotate_validation_set.py`; prompt construction and LLM calls are implemented in `LLMCommitAnnotator.py`. The disagreement analysis and metrics are implemented in `analysis/disagreement_analysis.ipynb`.

## Experimentation phase

- Main driver: `annotate_validation_set.py`
  - Reads a validation set of commits (default: `data/50-random-commits-validation.jsonl`).
  - Initializes `LLMCommitAnnotator` with flags: `--model`, `--context-mode`, `--temperature`, `--max-tokens`, and parallelism options.
  - Persists annotations under `data/llm-annotator-results/rX/` (one folder per run `r1..r5`).

- Prompt generation and LLM calls: `LLMCommitAnnotator.py`
  - Loads `documentation/definitions.md`, `documentation/context.md`, and optionally `documentation/few-shot-examples.md`.
  - Builds prompts according to `context_mode` flags: `message`, `diff`, `single-label`, `few-shot`, and their combinations.
  - Delegates model initialization and invocation to provider adapters implemented in the `llms/` package (Ollama, Copilot, OpenRouter, Google, OpenAI).
  - Expects each model response to be a JSON object containing `understanding`, `bfc`, `bpc`, `prc`, `nfc`, and `summary` fields.

- Provider adapters: `llms/` contains backend adapters (`ollama_llm.py`, `copilot_llm.py`, `openrouter_llm.py`, `google_llm.py`, `openai_llm.py`).

- Models executed (inferred from `data/llm-annotator-results` filenames):
  - `copilot/gpt-5-mini` (files labeled `annotations_copilot_gpt-5-mini.csv`)
  - CodeLlama variants: `codellama_34b`, `codellama_70b`
  - DeepSeek variants: `deepseek-coder-33b`, `deepseek-r1-32b`, `deepseek-r1-70b`
  - `gpt-oss` variants: `gpt-oss_20b`, `gpt-oss_120b`
  - `llama4_16x17b`
  - `qwen3-coder-30b`

  Results per model and per round are stored in `data/llm-annotator-results/r1..r5/` as CSV/JSON files.

- Rounds / context-mode configurations used in experiments:
  - `r1`: `message` (commit message only)
  - `r2`: `single-label`-> 
  - `r3`: `single-label+few-shot`
  - `r4`: `diff+single-label`
  - `r5`: `diff+single-label+few-shot`

- Reproducibility and configuration practices:
  - Default `temperature=0.0` for determinism where supported.
  - `max_tokens` increased to 10000 in the driver script to reduce truncated JSON outputs for verbose models.
  - The prompt enforces a compact output budget to reduce malformed JSON responses.

## Analysis phase

- Primary analysis notebook: `analysis/disagreement_analysis.ipynb`
  - Purpose: quantify agreement between LLM annotators and human annotators (A, B, C) using robust metrics.
  - Metrics used:
    - Krippendorff's Alpha (kA) — suitable for ordinal scores.
    - Cohen's Kappa (cK, quadratic) — pairwise agreement.
    - Alt-Test (aT) — probability that the LLM is closer to human consensus than a held-out human annotator.
  - Analysis flow (per LLM × round):
    1. Restrict to commits annotated by the three humans and the LLM.
    2. Compute the human baseline (kA, cK, aT) on the shared subset.
    3. Compute LLM metrics using leave-one-out permutations and averages as implemented in the notebook.
    4. Report Mean and Diff (Mean minus human baseline) for each metric.
    5. Repeat per label (`bfc`, `bpc`, `prc`, `nfc`) and aggregate into combined views.

- Supporting utilities:
  - `analysis/alt_test/alt_test.py` contains the Alt-Test routines used by the notebook.
  - The notebook relies on `pandas`, `seaborn`, `sklearn.metrics.cohen_kappa_score`, `krippendorff`, and helper functions to standardize CSVs.

## Artifacts and quick reproduction

- Annotation results by model and round: `data/llm-annotator-results/r1..r5/`.
- Human annotations: `data/human-annotator-results/annotations_A.csv`, `annotations_B.csv`, `annotations_C.csv`.
- Quick example to run a small experiment (r1) on the validation set:

```bash
python annotate_validation_set.py --input data/50-random-commits-validation.jsonl \
  --model "copilot/gpt-5-mini" --context-mode message --output output/
```

## Traceability and limitations

- The prompt template and single-label policy are defined in `LLMCommitAnnotator.py`.
- The experiment used multiple LLM backends; results depend on the adapter implementations in `llms/` and on local credentials or runtime configuration, which are not stored in the repository.
- Known issues to consider: malformed JSON responses from verbose models, calibration bias introduced by `few-shot` examples, and commit selection biases (merge commits were filtered when necessary).