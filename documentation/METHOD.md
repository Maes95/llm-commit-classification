# Methodology for Commit Annotation Using Large Language Models

In this document, the methodology used to evaluate the ability of Large Language Models (LLMs) to annotate software commits from the Linux Kernel repository will be described. 
The aim will be to link this study with the guidelines proposed by Baltes et al.[2], "Guidelines for Empirical Studies in Software Engineering involving Large Language Models":

- Declare LLM Usage and Role
- Report Model Version, Configuration, and Customizations
- Report Tool Architecture beyond Models
# Methodology (updated to reflect actual experiment and analysis)

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
  - `r2`: `single-label`
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

## Suggested next steps

- Record exact commit hashes used in each run for absolute reproducibility.
- Archive full prompts and raw responses (the `raw_response` field is already saved in per-commit outputs) in a separate audit artifact.
- Add automation to extract the notebook metrics into machine-readable CSV/JSON files for publication.

---
Updated based on repository code and data (scripts `annotate_validation_set.py`, `LLMCommitAnnotator.py`, adapters in `llms/`, files under `data/llm-annotator-results/`, and `analysis/disagreement_analysis.ipynb`).
    "presence_penalty": 0.0,   # No penalty for topic repetition
}
```

> **Note**: Guidelines: “[...] researchers may reduce the output variability by setting the temperature to a value close to 0 and setting a fixed seed value” 

### Phase 4: Prompt Validation and Refinement

Before annotating the full dataset, we implement an **Iterative Refinement Process** [1] to validate and optimize the prompt configuration. This methodology uses a validation subset to ensure high-quality annotations before full-scale deployment:

**Step 1 - Validation Sample Selection**: Select a random sample of 50 commits from the full dataset using a fixed pseudorandom seed (e.g., `seed=42`) to ensure reproducibility. No stratification by subsystem or commit type is performed to avoid introducing implicit bias or manual selection heuristics at this early validation stage. These 50 commits are reserved exclusively for prompt refinement and are excluded from all subsequent evaluation analyses to prevent data leakage.

**Step 2 - Initial LLM Annotation**: Apply the initial prompt (defined in Phase 2) to annotate the validation sample using each selected LLM model. Compare these LLM annotations against the existing human ground truth annotations for these 50 commits.

**Step 3 - Statistical Validation**: Calculate agreement coefficients, specifically Cohen's kappa (per dimension), between LLM and human annotations on the validation sample. This quantitatively assesses the alignment between LLM outputs and expected results, providing an objective measure of annotation quality.

**Step 4 - Criteria Assessment**: Evaluate whether the agreement coefficient meets a predefined criterion (Cohen's kappa > 0.5 per dimension, indicating strong agreement). If the criterion is met for all dimensions, proceed to Phase 5 with the validated prompt. If not, continue to Step 5 for prompt refinement.

> **Note**: Following the guidelines by Baltes et al. [2] regarding human validation of LLM outputs:
> "[...] If studies involve the annotation of software artifacts, and the goal is to automate the annotation process using LLM, researchers should follow systematic approaches to decide whether and how human annotators can be replaced. For example, Ahmed et al. suggest a method that involves using a jury of three LLMs with 3 to 4 few-shot examples rated by humans, where the model-to-model agreement on all samples is determined using Krippendorff’s Alpha. If the agreement is high (alpha > 0.5), a human rating can be replaced with an LLM-generated one. In cases of low model-to-model agreement (alpha ≤ 0.5), they then evaluate the prediction confidence of the model, selectively replacing annotations where the model confidence is high (≥ 0.8)” 
>  I believe that Cohen's Kappa and a limit of 0.5 could be an option, as cited in the Guidelines, but we may want to raise that number to 0.7 or 0.8.

**Step 5 - Error Analysis and Prompt Refinement**: When agreement falls below the threshold, perform detailed error analysis to identify systematic issues. Examine commits where LLM and human annotations diverge significantly (difference ≥2 points) to understand failure patterns. Systematically refine the prompt based on identified issues:

- **Simplifying language**: Use clearer, more straightforward language to improve LLM comprehension
- **Clarifying instructions**: Rewrite ambiguous sections to eliminate multiple interpretations
- **Adjusting context emphasis**: Modify instructions to better balance commit message, code changes, and metadata
- **Refining scoring rubric**: Add specific criteria or boundary cases to clarify score distinctions (e.g., when to assign 2 vs. 3)

**Step 6 - Iterative Repetition**: Return to Step 2 with the refined prompt and re-annotate the validation sample. Continue this cycle until achieving the target agreement level or reaching a maximum of 5 iterations. Document each iteration: prompt version, per-dimension kappa scores, specific modifications made, and rationale for changes.

Once the prompt achieves satisfactory agreement on the validation sample, the validated prompt configuration is finalized for full-scale deployment.

### Phase 5: Full Dataset Annotation

With the validated prompt from Phase 4, we proceed to annotate all 1,000 commits across all model-context combinations:

**Annotation Execution**: For each combination of model (10 models), context level (3 levels: minimal, standard, full), and commit (950 commits), we execute the LLM annotation process. For each annotation request, we systematically capture:

- **Execution time**: Wall-clock time (in seconds) from API request initiation to response completion, including network latency
- **Token consumption**: Total tokens used (input tokens + output tokens) as reported by the API provider's usage metadata
- **Timestamp**: ISO 8601 formatted timestamp of annotation completion for temporal analysis

These measurements enable performance benchmarking, cost estimation, and identification of potential bottlenecks or anomalies in the annotation process.

**Quality Assurance**: After completion, perform sanity checks on the annotation dataset:
- Verify all commits have annotations for all model-context combinations
- Check JSON parsing success rate (target: >99%)
- Validate score ranges (all scores 0-4)
- Identify and flag any anomalous patterns (e.g., identical annotations across many commits suggesting model errors)

The validation sample (50 commits from Phase 4) is excluded from all subsequent analyses to maintain methodological rigor and prevent overfitting to the validation set.

### Phase 6: Evaluation Metrics Computation

The full dataset of 1,000 commits (excluding the 50-commit validation sample) has been previously annotated by "expert" software engineers, establishing a gold standard ground truth dataset of 950 commits. Each commit has been manually scored across all four dimensions (BFC, BPC, PRC, NFC) using the same 0-4 rubric provided to the LLMs.

**Inter-Annotator Agreement Verification**: We first verify the reliability of ground truth annotations by calculating inter-annotator agreement metrics on commits that received multiple independent human annotations. This establishes the quality of the reference standard and provides an upper bound for expected LLM performance.

> **Note**: This work is partially complete. We have three annotation rounds: (1) individual annotations, (2) post-discussion annotations after resolving disagreements, and (3) consultation with a fourth annotator (Jesús) for remaining conflicts. Approximately 20 commits still have minor disagreements where annotators agree on primary dimensions (e.g., BFC=3-4) but differ on secondary dimensions (e.g., BPC scores vary 0-2).

**Systematic Metrics Computation**: For each model-context-dimension combination, we compute performance metrics by comparing LLM annotations against ground truth on the 950-commit evaluation set:

1. **Binary classification metrics** at two thresholds (strict: score ≥3; lenient: score ≥2): precision, recall, F1-score, and accuracy
2. **Score-level agreement**: Cohen's kappa (unweighted and quadratic-weighted), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE)
3. **Correlation measures**: Pearson and Spearman correlations between LLM and ground truth scores
4. **Percentage metrics**: Exact matches and within-1-point agreement rates

> **Note**: Some of the metrics are already suggested in the guidelines: “[...] For classification tasks, classical machine learning metrics such as Precision, Recall, F1-score, and Accuracy are often reported.”

This generates a comprehensive results dataset with metrics organized by model (10), context level (3), dimension (4), and metric type, facilitating multi-faceted performance analysis.

**Statistical Significance Testing**: We apply hypothesis tests to determine whether observed performance differences are statistically significant: McNemar's test for binary classification comparisons, Wilcoxon signed-rank test for pairwise score-level comparisons, and Friedman test with post-hoc Nemenyi tests for multi-model comparisons. All tests use α = 0.05 with Bonferroni correction for multiple comparisons.

Results are exported in structured CSV format containing all computed metrics, organized for subsequent analysis and visualization.

## Data Analysis

The analysis phase interprets the computed metrics to extract meaningful insights about LLM performance patterns, strengths, weaknesses, and practical implications:

### Quantitative Analysis

**Performance Comparison Across Conditions**: We analyze how different factors affect annotation quality by examining metric distributions across:
- **Model comparison**: Identify which LLM models achieve the highest agreement with human annotations
- **Context level impact**: Determine whether additional context (minimal vs. standard vs. full) significantly improves annotation accuracy
- **Dimension difficulty**: Identify which commit categories (BFC, BPC, PRC, NFC) are most challenging for LLMs to annotate correctly

**Score Distribution Analysis**: We characterize LLM annotation behavior by examining:
- **Central tendency and spread**: Compare mean scores and standard deviations between LLM and human annotations per dimension
- **Skewness and bias**: Identify systematic tendencies (e.g., over-scoring or under-scoring specific dimensions)
- **Score range utilization**: Assess whether LLMs use the full 0-4 scale or cluster around middle values
- **Confusion patterns**: Generate confusion matrices showing which score pairs (LLM vs. human) occur most frequently

**Multi-dimensional Patterns**: We explore relationships across annotation dimensions:
- **Dominant category identification**: For each commit, identify the highest-scored dimension and compare LLM vs. human agreement on primary category assignment
- **Dimension co-occurrence**: Calculate correlation matrices showing which dimensions tend to receive high scores together
- **Principal Component Analysis**: Identify underlying factors that explain variance in multi-dimensional annotations
- **Hierarchical clustering**: Group commits by their annotation profiles to discover natural commit categories

**Cost-Effectiveness Optimization**: We analyze operational trade-offs using the captured performance metrics:
- **Token consumption**: Total and per-commit token usage across model-context combinations, analyzing input vs. output token distribution
- **Execution time analysis**: Mean, median, and variance of annotation times per model-context combination to identify performance characteristics
- **API cost estimation**: Calculate monetary costs based on November-December 2025 provider pricing and measured token consumption
- **Throughput metrics**: Annotations per minute/hour accounting for actual measured latencies
- **Accuracy-cost frontier**: Identify Pareto-optimal configurations balancing performance metrics (F1-score, MAE) against computational expense (tokens, time, cost)

### Qualitative Analysis

**Reasoning Quality Assessment**: Using a sample of commits annotated by LLMs, we can evaluate:
- **Evidence grounding**: Whether reasoning explicitly cites commit message text, code changes, or metadata
- **Definition alignment**: Correct application of category definitions from the taxonomy
- **Logical coherence**: Internal consistency and sound argumentation structure
- **Score justification**: Clear connection between presented evidence and assigned score

## Bibliography

[1] V. De Martino, J. Castaño, F. Palomba, X. Franch and S. Martínez-Fernández, "A Framework for Using LLMs for Repository Mining Studies in Empirical Software Engineering," 2025 IEEE/ACM International Workshop on Methodological Issues with Empirical Studies in Software Engineering (WSESE), Ottawa, ON, Canada, 2025, pp. 6-11, doi: 10.1109/WSESE66602.2025.00008.

[2] Baltes, S., Angermeir, F., Arora, C., Barón, M. M., Chen, C., Böhme, L., ... & Wagner, S. (2025). Guidelines for Empirical Studies in Software Engineering involving Large Language Models. arXiv preprint arXiv:2508.15503.