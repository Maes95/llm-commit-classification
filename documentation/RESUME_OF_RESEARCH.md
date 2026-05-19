# Evaluating Large Language Models for Commit Annotation

## 1. Context and objective

The goal of this research is to evaluate whether large language models (LLMs) can annotate commits in a manner comparable to a human annotator.

As prior work, a manual annotation of 1,000 Linux kernel commits was carried out by three human annotators (A, B, C). Each commit was annotated across four dimensions using a score from 0 to 4 (0 = no, 4 = yes, with intermediate values for ambiguous cases; a single commit may receive non-zero scores in more than one category):
- **BFC** (Bug Fixing Commit): Does the commit fix a bug? (0–4)
- **BPC** (Bug Preventing Commit): Does the commit prevent a bug? (0–4)
- **PRC** (Perfective Commit): Does the commit improve quality without fixing or preventing a bug? (0–4)
- **NFC** (New Functionality Commit): Does the commit introduce new functionality? (0–4)

The experimentation process is divided into two phases: Experimentation and Analysis.

## 2. Experimentation phase

A subset of 50 commits was annotated using several LLMs and context configurations.

### 2.1 LLMs evaluated
- `gpt-5-mini`
- CodeLlama variants: `codellama_34b`, `codellama_70b`
- DeepSeek variants: `deepseek-coder-33b`, `deepseek-r1-32b`, `deepseek-r1-70b`
- `gpt-oss` variants: `gpt-oss_20b`, `gpt-oss_120b`
- `llama4_16x17b`
- `qwen3-coder-30b`
- `gemma4-31b`

### 2.2 Context configurations (rounds)
Each model was run under different context configurations to evaluate their impact on annotation quality:
- `r1`: Commit message only (`message`)
- `r2`: Single-dimension label (`single-label`)
- `r3`: `single-label` + examples (`few-shot`)
- `r4`: `diff` + `single-label`
- `r5`: `diff` + `single-label` + `few-shot`

For `few-shot` rounds, human-annotated commit examples were provided to guide the model. These examples were carefully selected to cover representative cases of full agreement, partial agreement, and full disagreement among human annotators, with the aim of improving the model's ability to handle ambiguous cases. They can be found in `documentation/few-shot-examples.md`.

### 2.3 Annotation process

Annotation was carried out using the `annotate_validation_set.py` script, which orchestrates reading the validation set, generating prompts, invoking the models, and persisting results to the `output/` folder.

#### 2.3.1 Model configuration

Each model was run with `temperature=0.0` to maximise determinism, and `max_tokens` was set to 10,000 to reduce the incidence of truncated JSON responses, particularly in more verbose models.

#### 2.3.2 Prompt design

The prompt included:
- The context provided to human annotators for their task, extracted from `documentation/context.md`.
- The definitions (taxonomy) of each dimension (BFC, BPC, PRC, NFC) given to humans, obtained from `documentation/definitions.md`.
- Specific instructions for the annotation task, including the scoring scale.
- Round-specific configurations:
  - **single-label**: Models were instructed to assign a score of 0 to 4 to a single dimension (the most relevant one) per commit, avoiding high scores across multiple dimensions for the same commit.
  - **few-shot**: Human-annotated commit examples were provided to guide the model, carefully selected to cover cases of full agreement, partial agreement, and full disagreement among human annotators, with the aim of improving handling of ambiguous cases. They can be found in `documentation/few-shot-examples.md`.
- The commit message without the "Fixes:" line, to avoid biasing the annotation towards BFC — consistent with the approach used for human annotators.
  - When the **diff** option was selected, it was included alongside the message, showing added and removed lines.
- Finally, the model was required to respond with a structured JSON containing scores for each dimension (`bfc`, `bpc`, `prc`, `nfc`), together with a brief reasoning, an overall assessment of its understanding of the commit (`understanding`), and a short commit summary (`summary`).

#### 2.3.3 Raw data processing

After experimentation, a set of raw annotations per model and configuration was obtained, stored in `output/rX/` as JSON files (one per commit/model/round). These files contain the full model response, including the reasoning and comprehension assessment.

To compare model annotations with human annotations, the JSON files were processed to extract only the scores assigned to each dimension (BFC, BPC, PRC, NFC) and consolidated into CSV files per model and round. These CSVs contain one row per commit with the scores assigned by the model, facilitating subsequent analysis. The `batch_convert_models_to_csv.py` script handled this processing and consolidation, generating the final CSV files found under `data/llm-annotator-results/rX/` for each model and round.

## 3. Analysis phase

In this phase, model annotations are compared against human annotations using inter-rater agreement metrics: Cohen's Kappa, Krippendorff's Alpha, and the Alt-Test.

The analysis phase is fully documented in the notebook `analysis/disagreement_analysis.ipynb` (it can be viewed online at [GitHub Pages](https://maes95.github.io/llm-commit-classification/)), which presents the agreement metric results for each model and configuration.

Although the notebook presents the agreement metric results for each model and configuration along with brief conclusions, the most relevant conclusions for each metric are repeated here, highlighting the models that approach or exceed human agreement. It is important to note that the results were calculated using the four categories (BFC, BPC, PRC, NFC), and an attempt was made to group the BFC and BPC annotations into a single category.

The results **without** grouping BFC and BPC are as follows:

- Based on metric Krippendorff's Alpha (kA): No LLM reaches the human inter-annotator agreement level (kA = 0.703). The best-performing LLM (**gpt-oss:20b**) achieves a kA Mean of 0.635 in round 3 (single-label + few-shot)
- Based on metric Cohen's Kappa (k): No LLM surpasses the human baseline (cK = 0.721). The best-performing LLM (**deepseek-r1:32b**) achieves a cK Mean of 0.659 in round 4 (diff + single-label).
- Based on Alt-Test:
  - When using rho, no LLM exceeds the per-run human baseline (all Diffs are negative), but the top models come remarkably close. The two best on average are **gpt-5-mini** (r2) and **gpt-oss:120b** (r3) (0.938).
  - Regarding Winning Rate (WR): **gpt-oss_120b** has the highest mean WR (0.917) in r3. This means that in 91.7% of the per-human comparisons across labels, the Alt-Test null hypothesis (LLM is not closer to human consensus than a human peer) is rejected after correction, suggesting that gpt-oss:120b's annotations are statistically indistinguishable from human agreement in those cases. 

The results **with** grouping BFC and BPC are as follows:

- Based on metric Krippendorff's Alpha (kA): No LLM reaches the human inter-annotator agreement level (kA = 0.739). The best-performing LLM (qwen3-coder_30b) achieves a kA Mean of 0.736 in round 5 (so close to human level)
- Based on metric Cohen's Kappa (k): At least 3 LLMs surpass or match the human baseline (cK = 0.751). The best-performing LLM (qwen3-coder_30b) achieves a cK Mean of 0.766 in round 5
- Based on Alt-Test:
  - When using rho, no LLM exceeds the per-run human baseline (all Diffs are negative), but the top models come remarkably close. The best on average is **gpt-oss:120b** (r3) with aT = 0.949 (beeing the baseline in this case 0.951).
  - Regarding Winning Rate (WR): gpt-oss_120b has the highest mean WR (0.889) in r3 and r4. This means that in 88.9% of the per-human comparisons across labels, the Alt-Test null hypothesis (LLM is not closer to human consensus than a human peer) is rejected after correction, suggesting that gpt-oss:120b's annotations are statistically indistinguishable from human agreement in those cases.

The results of human-to-human comparisons improve alongside those of LLMs (with the latter showing greater improvement on the kA and cK metrics). As for the AltTest results, the models that previously performed best continue to perform well, but with slightly lower scores; nevertheless, they remain viable candidates for replacing a human annotator.