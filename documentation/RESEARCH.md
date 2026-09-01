# RESEARCH.md — Consolidated reference of the study

> Quick-reference summary of the whole research project: what is being studied, how the
> experiments are run, how agreement is measured, and every numeric result currently stored
> in the repository. Written to be readable without re-running the notebook.
>
> Companion documents: [`METHOD.md`](METHOD.md) (paper-style methodology), [`RESUME_OF_RESEARCH.md`](RESUME_OF_RESEARCH.md)
> (short narrative summary), [`definitions.md`](definitions.md) + [`context.md`](context.md) (annotation guidelines given to
> humans *and* injected into every prompt), [`../README.md`](../README.md) (tool usage).
>
> Last synchronised with the repository: **2026-08-14** (working tree, branch `main`), after the
> notebook was re-executed with the 11-model selection restored (§6 numbers are the live notebook output;
> §7 numbers come from the previous execution, committed in `7b50f23`).

---

## Table of contents

1. [Research question and design](#1-research-question-and-design)
2. [Taxonomy and annotation rules](#2-taxonomy-and-annotation-rules)
3. [Datasets](#3-datasets)
4. [Experimentation phase (how annotations are produced)](#4-experimentation-phase-how-annotations-are-produced)
5. [Agreement metrics (as implemented)](#5-agreement-metrics-as-implemented)
6. [Study 1 — 11 LLMs × 5 rounds on 50 commits](#6-study-1--11-llms--5-rounds-on-50-commits)
7. [Study 2 — gpt-oss:120b × 5 rounds on 908 commits](#7-study-2--gpt-oss120b--5-rounds-on-908-commits)
8. [Cross-cutting findings](#8-cross-cutting-findings)
9. [State of the repository, caveats and known inconsistencies](#9-state-of-the-repository-caveats-and-known-inconsistencies)
10. [Reproduction recipes](#10-reproduction-recipes)
11. [Open questions / next steps](#11-open-questions--next-steps)

---

## 1. Research question and design

**Can an LLM replace a human annotator when labelling Linux-kernel commits by purpose?**

Two-phase design:

| Phase | What happens | Key artefacts |
|---|---|---|
| **Experimentation** | The same commits are annotated by several LLMs under several prompt/context configurations ("rounds") | [`LLMCommitAnnotator.py`](../LLMCommitAnnotator.py), [`annotate_validation_set.py`](../annotate_validation_set.py), `data/llm-annotator-results/rX/*.csv` |
| **Analysis** | LLM annotations are compared against three human annotators (A, B, C) with inter-rater agreement metrics | [`analysis/disagreement_analysis.ipynb`](../analysis/disagreement_analysis.ipynb), [`analysis/alt_test/alt_test.py`](../analysis/alt_test/alt_test.py) |

The unit of analysis is a single commit. The human trio is the *ceiling*: an LLM is "as good as a
human" if replacing one human with the LLM does not degrade group agreement, and if the LLM is
statistically not-worse than a held-out human at matching the consensus of the other two.

Published rendering of the notebook: https://maes95.github.io/llm-commit-classification/

---

## 2. Taxonomy and annotation rules

Four **independent** dimensions, each on a 5-point ordinal scale (0 = not applicable … 4 = primary characteristic):

| Label | Meaning |
|---|---|
| **BFC** — Bug-Fixing Commit | Fixes a fault that already manifests as a failure before the commit. Comment-only or pure-style changes cannot be BFC (behaviour is unchanged). |
| **BPC** — Bug-Preventing Commit | Prevents a *future/undiscovered* failure; no known bug is fixed (e.g. hardening a return value, defensive checks). |
| **PRC** — Perfective Commit | Improves quality (refactoring, performance, readability, comments) without fixing/preventing a bug or adding functionality. |
| **NFC** — New Feature Commit | Adds capability that did not exist (new hardware support, new API, new config option). |

Two rules that shape everything downstream:

- **Multi-label is allowed, but only for independent reasons.**
- **`Fixes:` lines are stripped** from every commit message (regex in `LLMCommitAnnotator._build_commit_context`)
  so neither humans nor models can shortcut to BFC via the kernel's convention tag.

Each annotator (human and LLM) also reports an **understanding** score (0–4) — a self-assessment of how
well the commit was comprehended — plus a free-text purpose/summary.

---

## 3. Datasets

Source: 1,000 randomly selected Linux-kernel commits (from `codeurjc/linux-bugs`), downloaded/reformatted by
[`data/generateData.py`](../data/generateData.py).

| File | Rows | Role |
|---|---|---|
| `data/1000-linux-commits.jsonl` | 1000 | Full corpus (includes merge commits) |
| `data/50-random-commits-validation.jsonl` | 50 | Validation subset, `random.seed(42)`, merge commits excluded ([`sample_random_commits.py`](../data/sample_random_commits.py)) |
| `data/50-random-commits-validation-with-diff.jsonl` | 50 | Same + unified diff, stats, file list |
| `data/858-linux-commits.jsonl` | 858 | The remainder (corpus minus merges, minus the 50, minus 3 manually excluded hashes — see [`filter_unselected_commits.py`](../data/filter_unselected_commits.py)) |
| `data/858-linux-commits-with-diff.jsonl` | 858 | Same + diff/stats/files (built with [`utils/add_diff_to_jsonl.py`](../utils/add_diff_to_jsonl.py) against a local kernel clone) |

Human annotations (`data/human-annotator-results/annotations_{A,B,C}.csv`, annotators Michel / Abhishek / David):
A = 911 rows, B = 1003 rows, C = 911 rows; **911 commits are annotated by all three**.

Two effective evaluation sets appear in the results:

- **50 commits** — the intersection humans ∩ every LLM in Study 1.
- **908 commits** — the intersection humans ∩ the large gpt-oss:120b run (858 + 50) in Study 2.

---

## 4. Experimentation phase (how annotations are produced)

### 4.1 Prompt assembly (`LLMCommitAnnotator._build_prompt`)

Fixed layer order, identical for every model:

1. System instruction (expert commit-annotation analyst).
2. `documentation/context.md` — the exact context given to human annotators, including the priority rule.
3. `documentation/definitions.md` — the taxonomy.
4. Understanding rubric (0–4) + scoring rubric (0–4).
5. `[SINGLE-LABEL POLICY]` block *(only when the `single-label` flag is on)*: by default exactly one
   category may be > 0; a second is allowed only if `understanding.score <= 2` **or** there is explicit,
   independent, comparably strong evidence of two purposes.
6. `[FEW-SHOT HUMAN EXAMPLES]` block *(only when `few-shot` is on)*: content of
   [`documentation/few-shot-examples.md`](few-shot-examples.md) — 3 real commits with the per-annotator
   scores and rationales of A, B and C, chosen as *total agreement*, *partial disagreement* and *high disagreement*.
7. `[OUTPUT BUDGET]` — word caps (80 / 60 / 40 words) to curb verbosity and truncated JSON.
8. Commit context: message (with `Fixes:` removed); plus diff + stats + modified files when `diff` is on.
9. Strict raw-JSON output contract (`understanding`, `bfc`, `bpc`, `prc`, `nfc`, `summary`).

### 4.2 Context modes and rounds

Modes are composable flags joined with `+`: `message` (base), `diff`, `single-label`, `few-shot`.
The five rounds actually run ([`experiments/runAll.sh`](../experiments/runAll.sh)):

| Round | `--context-mode` | Prompt contains |
|---|---|---|
| **r1** | `message` | message only |
| **r2** | `single-label` | message + single-label policy |
| **r3** | `single-label+few-shot` | r2 + human examples |
| **r4** | `diff+single-label` | message + diff/stats/files + single-label policy |
| **r5** | `diff+single-label+few-shot` | everything |

Note: `diff` is never evaluated without `single-label`, so "effect of the diff" is only observable as r2→r4 and r3→r5.

### 4.3 Execution

- `temperature = 0.0`, `max_tokens = 10000` (raised from 3072 because verbose models truncated their JSON).
- Providers auto-routed by model id (`llms/`): Ollama (local), GitHub Copilot, OpenRouter, Google, OpenAI.
- Batch driver: thread pool (`--workers`, 10 in the scripts), rate-limit retry (`--max-retries 3`, `--retry-delay 90`),
  **resume-safe** (a commit whose `{hash}.json` already exists is skipped).
- One JSON per commit under `output/rX/{model}/`, containing the parsed scores, reasonings, timing,
  token usage, the raw response *and* the full prompt.
- Local/GPU runs were launched on the URJC SLURM cluster ([`experiments/run.sh`](../experiments/run.sh), `slurm_run_experiment.sh`).
- Post-processing to CSV: [`utils/batch_convert_models_to_csv.py`](../utils/batch_convert_models_to_csv.py) →
  `data/llm-annotator-results/rX/annotations_{provider}_{model}.csv`, one row per commit with the four scores
  (same column layout as the human CSVs, so the notebook loads both identically).

### 4.4 Models evaluated (Study 1)

`gpt-5-mini` (GitHub Copilot API) and, on local Ollama: `codellama:34b`, `codellama:70b`, `deepseek-coder:33b`,
`deepseek-r1:32b`, `deepseek-r1:70b`, `gpt-oss:20b`, `gpt-oss:120b`, `llama4:16x17b`, `qwen3-coder:30b`, `gemma4:31b`.

---

## 5. Agreement metrics (as implemented)

All metrics are computed **per label** and then averaged into an "overall" figure; each LLM is evaluated on
the commits it shares with all three humans.

### 5.1 Krippendorff's Alpha — `kA`
`krippendorff.alpha(..., level_of_measurement="ordinal")`.
- Human baseline: `kA(A,B,C)`.
- LLM: mean over the three **leave-one-out replacements** `{(LLM,B,C), (A,LLM,C), (A,B,LLM)}`.
- `kA Diff = kA Mean − kA(A,B,C)`.

### 5.2 Cohen's Kappa — `cK`
`cohen_kappa_score(..., weights="quadratic")` (undefined → NaN when both vectors are constant).
- Human baseline: mean of `cK(A,B)`, `cK(A,C)`, `cK(B,C)`.
- LLM: mean of `cK(A,LLM)`, `cK(B,LLM)`, `cK(C,LLM)`.
- `cK Diff = cK Mean − human baseline`.

### 5.3 Alt-Test — `aT` (`analysis/alt_test/alt_test.py`)
Leave-one-human-out substitutability test:
- Alignment score = **negative RMSE** of a rating against the two remaining humans
  (rank-equivalent to distance from their mean).
- For each excluded human *h<sub>j</sub>* and each commit: `W_f = 1 if S(LLM) >= S(h_j)`, `W_h = 1 if S(h_j) >= S(LLM)`
  — **ties credit both sides**, so `ρ_f + ρ_h ≥ 1`.
- `ρ_f` (reported as **aT Mean**) = mean of `W_f`; `ρ_h` (reported as **Baseline (rho_h)**) = mean of `W_h`.
- Hypothesis test per excluded human on `d = W_h − W_f`, H₁: `E[d] < ε` with **ε = 0.15**
  (cost-benefit discount for "skilled annotators"), α = 0.05, **Benjamini–Yekutieli** FDR correction across the 3 tests.
- **WR (Winning Rate)** = fraction of the 3 tests rejected, averaged over labels
  (per label ∈ {0, ⅓, ⅔, 1}). `WR ≥ 0.5` ⇒ `can_replace = True`.
- **aT Diff = aT Mean − that row's own `rho_h`**, i.e. against the human reference computed inside the same
  run — *not* against the fixed "Humans (baseline)" row.

### 5.4 Analysis variants
- **Original labels**: BFC, BPC, PRC, NFC.
- **Merged labels**: `BFC_merged = max(BFC, BPC)`, BPC dropped → BFC, PRC, NFC. Rationale: BFC and BPC are the
  reactive/proactive faces of the same bug-related activity, and BPC is by far the least reliable dimension
  even among humans.

---

## 6. Study 1 — 11 LLMs × 5 rounds on 50 commits

### 6.1 Human baseline (50 commits)

| Label | kA(A,B,C) | cK(A,B) | cK(A,C) | cK(B,C) | cK mean | aT (human) |
|---|---|---|---|---|---|---|
| bfc | 0.842 | 0.866 | 0.832 | 0.872 | 0.857 | 0.953 |
| bpc | **0.448** | 0.299 | 0.432 | 0.656 | **0.463** | 0.880 |
| prc | 0.723 | 0.635 | 0.731 | 0.815 | 0.727 | 0.893 |
| nfc | 0.798 | 0.770 | 0.858 | 0.888 | 0.838 | 0.967 |
| **overall** | **0.703** | 0.643 | 0.713 | 0.808 | **0.721** | **0.923** |

Merged labels: BFC_merged kA 0.697 / cK 0.689 / aT 0.893 → **overall kA 0.739, cK 0.751, aT 0.918**.

Humans themselves disagree most on BPC (kA 0.448) and the A–B pair is the weakest (cK 0.643).

### 6.2 Krippendorff's Alpha — kA Mean (baseline 0.703)

| Model | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| codellama_34b | 0.427 | 0.437 | 0.330 | 0.405 | 0.270 |
| codellama_70b | 0.374 | 0.383 | 0.138 | 0.207 | 0.209 |
| deepseek-coder_33b | 0.414 | 0.402 | 0.384 | 0.343 | 0.281 |
| deepseek-r1_32b | 0.567 | 0.600 | 0.601 | **0.629** | 0.611 |
| deepseek-r1_70b | 0.537 | 0.607 | 0.577 | 0.573 | 0.599 |
| gemma4_31b | 0.578 | 0.592 | 0.596 | 0.599 | 0.607 |
| gpt-5-mini | 0.608 | 0.606 | 0.577 | 0.562 | 0.558 |
| gpt-oss_120b | 0.624 | 0.561 | 0.608 | 0.582 | 0.572 |
| gpt-oss_20b | 0.601 | 0.578 | **0.635** | 0.549 | 0.592 |
| llama4_16x17b | 0.557 | 0.495 | 0.472 | 0.435 | 0.344 |
| qwen3-coder_30b | 0.580 | 0.608 | 0.571 | 0.622 | 0.582 |

**Best: gpt-oss:20b r3 = 0.635 (Δ −0.068).** No model reaches the human level.

### 6.3 Cohen's Kappa — cK Mean (baseline 0.721)

| Model | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| codellama_34b | 0.302 | 0.379 | 0.296 | 0.322 | 0.195 |
| codellama_70b | 0.249 | 0.230 | 0.058 | −0.014 | 0.024 |
| deepseek-coder_33b | 0.283 | 0.302 | 0.205 | 0.166 | 0.120 |
| deepseek-r1_32b | 0.595 | 0.568 | 0.563 | **0.659** | 0.596 |
| deepseek-r1_70b | 0.545 | 0.605 | 0.516 | 0.574 | 0.563 |
| gemma4_31b | 0.528 | 0.536 | 0.546 | 0.549 | 0.558 |
| gpt-5-mini | 0.545 | 0.572 | 0.528 | 0.505 | 0.486 |
| gpt-oss_120b | 0.603 | 0.484 | 0.573 | 0.550 | 0.510 |
| gpt-oss_20b | 0.561 | 0.520 | 0.589 | 0.490 | 0.537 |
| llama4_16x17b | 0.510 | 0.420 | 0.375 | 0.247 | 0.223 |
| qwen3-coder_30b | 0.536 | 0.588 | 0.519 | 0.595 | 0.542 |

**Best: deepseek-r1:32b r4 = 0.659 (Δ −0.062).** No model reaches the human level.

### 6.4 Alt-Test — aT Mean (WR in parentheses); human reference ≈ 0.92–0.96

| Model | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| codellama_34b | 0.687 (0.00) | 0.660 (0.17) | 0.617 (0.00) | 0.667 (0.25) | 0.580 (0.00) |
| codellama_70b | 0.592 (0.17) | 0.615 (0.00) | 0.435 (0.00) | 0.492 (0.00) | 0.500 (0.00) |
| deepseek-coder_33b | 0.582 (0.00) | 0.580 (0.00) | 0.750 (0.00) | 0.568 (0.00) | 0.575 (0.00) |
| deepseek-r1_32b | 0.670 (0.25) | 0.812 (0.25) | 0.822 (0.25) | 0.815 (**0.50**) | 0.867 (0.42) |
| deepseek-r1_70b | 0.703 (0.25) | 0.808 (0.25) | 0.863 (0.25) | 0.810 (0.25) | 0.865 (0.42) |
| gemma4_31b | 0.888 (0.17) | 0.900 (0.25) | 0.913 (**0.58**) | 0.915 (0.25) | 0.910 (0.25) |
| gpt-5-mini | 0.870 (**0.50**) | **0.938** (**0.67**) | 0.918 (**0.67**) | 0.918 (0.42) | 0.898 (0.42) |
| gpt-oss_120b | 0.853 (0.25) | 0.893 (**0.50**) | **0.938** (**0.92**) | 0.923 (**0.75**) | 0.913 (0.42) |
| gpt-oss_20b | 0.835 (0.25) | 0.882 (0.17) | 0.892 (**0.58**) | 0.877 (0.25) | 0.882 (0.17) |
| llama4_16x17b | 0.640 (0.00) | 0.802 (0.00) | 0.745 (0.00) | 0.705 (0.00) | 0.593 (0.00) |
| qwen3-coder_30b | 0.707 (0.25) | 0.845 (0.25) | 0.855 (0.25) | 0.868 (0.25) | 0.885 (0.25) |

All aT Diffs are negative (no LLM beats the held-out human), but **gpt-oss:120b r3 reaches WR = 0.917**:
in 91.7 % of the per-human tests the null "the LLM is not closer to the consensus than a human peer" is rejected
after BY correction. Models with WR ≥ 0.5 (the `can_replace` threshold): gpt-oss_120b (r2/r3/r4),
gpt-5-mini (r1/r2/r3), gemma4_31b (r3), gpt-oss_20b (r3), deepseek-r1_32b (r4).

**Where the WR of the best model comes from** (notebook cell 36, per-label WR for gpt-oss:120b):

| Round | bfc | bpc | nfc | prc | mean (reported WR) |
|---|---|---|---|---|---|
| r1 | 0.000 | 0.000 | 1.000 | 0.000 | 0.250 |
| r2 | 0.000 | 0.667 | 0.667 | 0.667 | 0.500 |
| r3 | **1.000** | 0.667 | 1.000 | 1.000 | **0.917** |
| r4 | 0.000 | 1.000 | 1.000 | 1.000 | 0.750 |
| r5 | 0.000 | 0.667 | 1.000 | 0.000 | 0.417 |

Note the asymmetry: **BFC is the label where the model is least substitutable** — WR = 0 in four of the five
rounds, and r3 is the only configuration where it also wins on BFC. NFC is the easy label (WR = 1.0 everywhere).
Aggregate WR therefore hides the fact that most of the "replaceability" comes from the labels that matter least
for a bug-related study.

### 6.5 Merged labels (BFC = max(BFC, BPC)) — baselines kA 0.739 / cK 0.751 / aT 0.918

kA Mean:

| Model | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| codellama_34b | 0.549 | 0.497 | 0.295 | 0.461 | 0.258 |
| codellama_70b | 0.441 | 0.465 | 0.120 | 0.236 | 0.242 |
| deepseek-coder_33b | 0.512 | 0.444 | 0.473 | 0.412 | 0.319 |
| deepseek-r1_32b | 0.637 | 0.710 | 0.711 | 0.704 | 0.715 |
| deepseek-r1_70b | 0.632 | 0.683 | 0.673 | 0.668 | 0.711 |
| gemma4_31b | 0.649 | 0.648 | 0.670 | 0.688 | 0.689 |
| gpt-5-mini | 0.641 | 0.674 | 0.665 | 0.666 | 0.699 |
| gpt-oss_120b | 0.703 | 0.676 | 0.731 | 0.682 | 0.685 |
| gpt-oss_20b | 0.700 | 0.682 | 0.709 | 0.684 | 0.697 |
| llama4_16x17b | 0.614 | 0.598 | 0.601 | 0.497 | 0.449 |
| qwen3-coder_30b | 0.654 | 0.712 | 0.715 | 0.723 | **0.736** |

cK Mean:

| Model | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| codellama_34b | 0.396 | 0.413 | 0.211 | 0.363 | 0.172 |
| codellama_70b | 0.294 | 0.362 | −0.002 | −0.000 | 0.042 |
| deepseek-coder_33b | 0.444 | 0.360 | 0.321 | 0.250 | 0.160 |
| deepseek-r1_32b | 0.688 | 0.743 | 0.741 | **0.754** ✅ | 0.735 |
| deepseek-r1_70b | 0.648 | 0.705 | 0.660 | 0.698 | 0.723 |
| gemma4_31b | 0.638 | 0.613 | 0.648 | 0.678 | 0.674 |
| gpt-5-mini | 0.598 | 0.675 | 0.659 | 0.660 | 0.681 |
| gpt-oss_120b | 0.719 | 0.649 | **0.751** ✅ | 0.674 | 0.678 |
| gpt-oss_20b | 0.689 | 0.649 | 0.686 | 0.675 | 0.666 |
| llama4_16x17b | 0.597 | 0.581 | 0.545 | 0.338 | 0.367 |
| qwen3-coder_30b | 0.632 | 0.740 | 0.724 | 0.715 | **0.766** ✅ |

✅ = meets or exceeds the human baseline (0.751). **qwen3-coder:30b r5 = 0.766 is the only clear pass.**

aT Mean (WR):

| Model | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| codellama_34b | 0.667 (0.00) | 0.633 (0.22) | 0.562 (0.00) | 0.640 (0.33) | 0.547 (0.00) |
| codellama_70b | 0.484 (0.22) | 0.529 (0.00) | 0.413 (0.00) | 0.420 (0.00) | 0.480 (0.00) |
| deepseek-coder_33b | 0.500 (0.00) | 0.467 (0.00) | 0.749 (0.22) | 0.473 (0.00) | 0.540 (0.00) |
| deepseek-r1_32b | 0.638 (0.33) | 0.816 (0.33) | 0.833 (0.33) | 0.791 (0.33) | 0.873 (**0.56**) |
| deepseek-r1_70b | 0.696 (0.33) | 0.789 (0.33) | 0.876 (0.33) | 0.789 (0.33) | 0.884 (**0.56**) |
| gemma4_31b | 0.878 (0.22) | 0.882 (0.33) | 0.896 (0.22) | 0.909 (0.33) | 0.902 (0.33) |
| gpt-5-mini | 0.844 (0.33) | 0.922 (0.44) | 0.909 (0.44) | 0.922 (**0.56**) | 0.916 (**0.56**) |
| gpt-oss_120b | 0.849 (0.33) | 0.889 (0.44) | **0.949** (**0.89**) | 0.929 (**0.89**) | 0.922 (**0.67**) |
| gpt-oss_20b | 0.840 (0.33) | 0.893 (0.22) | 0.902 (**0.67**) | 0.907 (**0.67**) | 0.900 (0.33) |
| llama4_16x17b | 0.600 (0.00) | 0.773 (0.00) | 0.762 (0.00) | 0.656 (0.00) | 0.602 (0.00) |
| qwen3-coder_30b | 0.689 (0.33) | 0.847 (0.33) | 0.869 (**0.67**) | 0.869 (0.33) | 0.916 (**0.67**) |

**gpt-oss:120b r3 = 0.949 vs a human reference of 0.951** — essentially indistinguishable.

### 6.6 Study 1 takeaways

- With the four original labels, **no configuration matches human agreement** on kA or cK; the gap of the best
  models is ≈ −0.07 (kA) / −0.06 (cK).
- **Merging BFC+BPC closes most of the gap**: the human baseline itself rises (kA 0.703→0.739, cK 0.721→0.751)
  but the LLMs rise faster, and 3 configurations meet/beat the cK baseline. BPC is the bottleneck dimension.
- On the substitutability test, the top models (gpt-oss:120b, gpt-5-mini, gpt-oss:20b, qwen3-coder:30b,
  deepseek-r1) cross `can_replace = True` in several rounds.
- **Code-specialised models are the worst annotators**: codellama:34b/70b and deepseek-coder:33b collapse
  (cK as low as −0.014); bigger is not better (codellama:70b < codellama:34b). General-purpose reasoning
  models dominate.
- **No single round wins for everyone.** r3 (single-label + few-shot) favours the gpt-oss family; r4
  (diff + single-label) favours deepseek-r1 and qwen3-coder; r5 helps qwen3-coder but hurts gpt-5-mini and
  the code models. Extra context is not monotonically useful.

---

## 7. Study 2 — gpt-oss:120b × 5 rounds on 908 commits

The **scale-up run**: the best Alt-Test model of Study 1 annotating the whole human-annotated corpus
(858 + 50 = **908 commits**), from `data/llm-annotator-results/rX/annotations_ollama_gpt-oss_120b_858.csv`.

These numbers are *not* in the notebook right now — they are the outputs committed in `7b50f23`
("Add last results to repo"). To regenerate them, invert the selection in cell 4: uncomment
`"gpt-oss_120b_858"` and comment the other 11 entries (see the boxed note in §6 for why the two sets must
not be selected together).

### 7.1 Human baseline (908 commits)

| Label | kA(A,B,C) | cK(A,B) | cK(A,C) | cK(B,C) | cK mean | aT (human) |
|---|---|---|---|---|---|---|
| bfc | 0.836 | 0.833 | 0.858 | 0.855 | 0.849 | 0.943 |
| bpc | **0.543** | 0.351 | 0.512 | 0.706 | **0.523** | 0.885 |
| prc | 0.807 | 0.718 | 0.795 | 0.881 | 0.798 | 0.914 |
| nfc | 0.830 | 0.807 | 0.872 | 0.912 | 0.864 | 0.957 |
| **overall** | **0.754** | 0.677 | 0.759 | 0.838 | **0.758** | **0.924** |

Merged labels: BFC_merged 0.743 / 0.738 / 0.892 → **overall kA 0.793, cK 0.800, aT 0.921**.
Human agreement is slightly *higher* on the large set than on the 50-commit sample, so baselines are stricter.

### 7.2 Results — original labels (baselines kA 0.754, cK 0.758)

| Round | kA Mean (Δ) | cK Mean (Δ) | aT Mean | aT rho_h | aT Δ | WR |
|---|---|---|---|---|---|---|
| r1 `message` | 0.667 (−0.087) | 0.646 (−0.113) | 0.834 | 0.916 | −0.082 | 0.750 |
| r2 `single-label` | 0.665 (−0.089) | 0.631 (−0.127) | 0.916 | 0.957 | −0.041 | **1.000** |
| r3 `+few-shot` | 0.644 (−0.109) | 0.602 (−0.156) | 0.911 | 0.957 | −0.046 | **1.000** |
| r4 `diff+single-label` | 0.639 (−0.115) | 0.592 (−0.166) | 0.907 | 0.957 | −0.051 | **1.000** |
| r5 `all` | 0.627 (−0.127) | 0.573 (−0.185) | 0.901 | 0.958 | −0.057 | **1.000** |

### 7.3 Results — merged labels (baselines kA 0.793, cK 0.800)

| Round | kA Mean (Δ) | cK Mean (Δ) | aT Mean | aT rho_h | aT Δ | WR |
|---|---|---|---|---|---|---|
| r1 | 0.750 (−0.043) | 0.759 (−0.041) | 0.838 | 0.908 | −0.070 | 0.667 |
| r2 | **0.767 (−0.026)** | **0.763 (−0.037)** | 0.932 | 0.951 | −0.019 | **1.000** |
| r3 | 0.763 (−0.030) | 0.757 (−0.043) | **0.934** | 0.950 | **−0.016** | **1.000** |
| r4 | 0.749 (−0.044) | 0.735 (−0.065) | 0.925 | 0.951 | −0.026 | **1.000** |
| r5 | 0.748 (−0.045) | 0.733 (−0.067) | 0.926 | 0.951 | −0.025 | **1.000** |

### 7.4 Study 2 takeaways

- The picture **survives the 18× scale-up in direction but not in magnitude**: on 908 commits gpt-oss:120b stays
  0.03–0.13 below the human baselines on kA/cK, whereas on 50 commits its merged-label cK (0.751 in r3) had
  touched the baseline. Small-sample optimism in Study 1 is real.
- **WR = 1.000 in r2–r5 (both label sets)**: with ~900 commits the hypothesis test has enough power that every
  per-human comparison rejects the null. WR becomes uninformative at this sample size — it saturates — while
  `aT Diff` stays negative (−0.016 … −0.057). Report ρ_f/Δ, not WR alone, on the large set.
- **Extra context degrades the ordinal metrics monotonically here**: kA/cK are best in r1–r2 and worst in r5.
  Few-shot and diff do not pay off for this model at scale.
- The **single-label policy is what buys the Alt-Test gain** (r1→r2: aT 0.834→0.916, WR 0.75→1.00) while
  costing a little on cK — it makes the model's dominant label sharper but suppresses legitimate secondary labels.

---

## 8. Cross-cutting findings

1. **BPC is the weakest dimension for everyone.** Human kA on BPC is 0.448 (50) / 0.543 (908) versus 0.80+ for
   BFC/PRC/NFC; human cK(A,B) on BPC is only 0.299/0.351. Any headline "LLM ≈ human" claim depends on how BPC
   is handled, which is exactly why the merged-label variant exists.
2. **The single-label policy is over-obeyed.** Measured on the 908-commit run, the share of commits with more
   than one non-zero label is: humans 20.6–24.3 %, gpt-oss:120b **47.4 % in r1 → 0.2–0.4 % in r2–r5**.
   The instruction converts a *soft priority rule* into a *hard exclusivity rule*.
3. **Consequence: BPC nearly disappears.** Share of commits with BPC > 0 — humans 19.3 / 29.1 / 26.5 %;
   gpt-oss:120b 34.8 % (r1) → 5.2 / 4.0 / 4.8 / 3.9 % (r2–r5). The model does not *disagree* about BPC so much as
   it *stops emitting* it, which mechanically depresses per-label kA/cK and is largely repaired by merging BFC+BPC.
4. **PRC is over-assigned in the free mode**: 71.1 % of commits get PRC > 0 in r1 vs 55–61 % for humans; the
   single-label policy pulls this to ~46 %.
5. **Weak models ignore the policy.** In r5 the multi-label rate is 0 % for gpt-5-mini/gemma4/gpt-oss:120b but
   still 62–74 % for codellama:70b/34b and deepseek-coder — instruction-following capacity, not size, separates
   good annotators from bad.
6. **The diff rarely helps.** r2→r4 and r3→r5 are neutral-to-negative for most models on kA/cK; only
   deepseek-r1:32b and qwen3-coder:30b benefit clearly (Study 1). Cost (tokens, runtime) rises substantially.
7. **Metric disagreement is informative.** Alt-Test rewards models that sit close to the consensus on the
   *many easy commits*; kA/cK punish systematic label-distribution shifts. gpt-oss:120b is Alt-Test-excellent
   and kA-mediocre — i.e. it is a plausible *individual* annotator but changes the *marginal distribution* of labels.
8. **Averaged WR hides where the model actually wins.** For gpt-oss:120b (§6.4) NFC has WR = 1.0 in every round
   while BFC has WR = 0.0 in four of five; the headline 0.917 in r3 is the only configuration that also
   passes on BFC. Substitutability should be read per label, not as a single mean.

---

## 9. State of the repository, caveats and known inconsistencies

Facts to keep in mind before quoting numbers from this repo:

- **Two different studies share one notebook, switched by `SELECTED_LLMS` in cell 4.** The working tree now
  renders **Study 1** (11 models, 50 commits) and its prose conclusions in sections 8.x/9.x match the tables
  again, as does [`RESUME_OF_RESEARCH.md`](RESUME_OF_RESEARCH.md). **Study 2** (gpt-oss:120b, 908 commits) is
  only preserved in commit `7b50f23` and in §7 of this document — re-running Study 1 overwrote those cells.
  If both runs are to be kept side by side, they need either two notebooks or an exported results file;
  right now each execution destroys the other's outputs.
- **Whichever run is rendered, the git diff of the notebook is huge** (all styled HTML tables are re-emitted).
  Expect ~1k changed lines per execution even when the numbers are identical.
- **METHOD.md §3.6.3 does not match the code.** It describes a "one-sided bootstrap hypothesis test with 10,000
  resamples", but the notebook calls `perform_alt_test(..., epsilon=0.15)` and the default is
  `hypothesis_test="t_test"` (one-sample `ttest_1samp(d, ε, alternative="less")`). The bootstrap path exists in
  `alt_test.py` but is not the one used. METHOD.md also writes ρ with a ½-weight for ties, whereas the
  implementation gives ties full credit to *both* sides.
- **METHOD.md lists `gemma4:31b` and `llama4:16×17b`**; these are the identifiers used in the run scripts and CSV
  names. Verify against the real Ollama tags before publication.
- **`data/llm-annotator-resultsNEW/`** contains an r1-only set for 9 models and is not read by the notebook
  (`LLM_ROOT` points at `data/llm-annotator-results`). Purpose undocumented — treat as scratch until confirmed.
- **`analysis/disagreement_analysisOLD.ipynb`** is a superseded iteration with a different model set
  (gemma3:12b, llama3.1:8b/70b, mistral:7b, Hermes-4.3-36b, deepseek-r1:14b) and only r1/r2; its human baseline
  is kA 0.703. Do not mix its numbers with §6. `analysis/llm_replacement_analysis.ipynb` is the Alt-Test
  derivation/ε-sensitivity notebook (source of `alt_test.py`).
- **`analysis/figures/*.png`** (confusion matrices, Krippendorff plot) come from the older notebooks — they are
  not regenerated by the current one.
- **The paper skeleton `2026-LLMCommitAnnotations/` is empty**: `results.tex`, `discussion.tex`,
  `conclusions.tex`, `introduction.tex` etc. are `TODO`. METHOD.md is the only written methodology.
- **Uncommitted work**: `documentation/METHOD.md` and `analysis/disagreement_analysis.ipynb` (the Study 1
  re-execution) are modified in the working tree relative to `HEAD` (`7b50f23`).
- Raw per-commit JSON under `output/` is git-ignored; only the derived CSVs are versioned. The JSONs hold the
  reasoning text, the token usage and the exact prompt — needed for any qualitative/error analysis.

---

## 10. Reproduction recipes

```bash
# 0) environment
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp dotenv-example .env          # only needed for hosted providers

# 1) (re)build the diff-enriched dataset
python utils/add_diff_to_jsonl.py data/858-linux-commits.jsonl \
       data/858-linux-commits-with-diff.jsonl --repo /path/to/linux

# 2) run the five rounds for one model (r1..r5, resume-safe)
./experiments/runAll.sh "ollama/gpt-oss:120b"        # writes output/rX/<model>/*.json
#    on the SLURM cluster:  ./experiments/run.sh "gpt-oss:120b" "H100:1"

# 3) consolidate to CSV (skips already-converted files unless --force)
python utils/batch_convert_models_to_csv.py --output-base data/llm-annotator-results

# 4) analysis — cell 4 selects which study is rendered (filter=True is always on)
jupyter lab analysis/disagreement_analysis.ipynb
#    - Study 1 (current): the 11 model names listed, "gpt-oss_120b_858" commented out  → 50 commits
#    - Study 2:           only "gpt-oss_120b_858" uncommented, the 11 commented out    → 908 commits
#    Never select both groups at once: each LLM is scored on its own commit intersection
#    while the human baseline row uses the intersection of all selected files (= 50).
```

Single commit, for prompt debugging:

```bash
python annotate_simple.py data/sample-commits/724-*.json \
       --model "ollama/gpt-oss:20b" --context-mode "diff+single-label+few-shot"
```

---

## 11. Open questions / next steps

- **Finish the scale-up matrix.** Study 2 has one model. The Study 1 leaders (gpt-5-mini, qwen3-coder:30b,
  deepseek-r1:32b, gpt-oss:20b) on the 908-commit set would show whether the 50-commit ranking is stable —
  currently the strongest claims rest on n = 50.
- **Report a sample-size-robust substitutability statistic.** WR saturates at 1.000 on 908 commits; consider
  reporting ρ_f − ρ_h with confidence intervals, or an ε-sensitivity curve (the ε grid already exists in
  `llm_replacement_analysis.ipynb`). Report it **per label** too — the aggregate hides that BFC almost never wins.
- **Persist the metrics instead of the notebook outputs.** `results_df` / `merged_results_df` dumped to CSV
  (one file per run configuration) would let Study 1 and Study 2 coexist, make the notebook diffs small, and
  remove the need to dig numbers out of git history as this document had to.
- **Reconcile METHOD.md with `alt_test.py`** (t-test vs bootstrap, tie handling) before writing `results.tex`.
- **Soften the single-label policy.** The current wording drives the multi-label rate to ~0 % versus ~22 % in
  humans. A "priority-ordered, secondary labels allowed when independently justified" phrasing would test the
  guideline rather than an exclusivity rule.
- **Isolate the diff factor.** There is no `diff`-only round; `message` vs `diff` at constant policy is not measurable.
- **Use the `understanding` score.** It is collected from every model and every human and currently unused —
  natural for a difficulty-stratified analysis (do LLM and human agreement fall on the same commits?).
- **Qualitative error analysis** on the reasoning fields of the stored JSONs, especially the BFC/BPC confusions
  that the merged-label variant papers over.
