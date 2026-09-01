# Methodology

## 3.1 Study Design

This study evaluates whether Large Language Models (LLMs) can annotate software commits from the Linux Kernel repository at a level of reliability comparable to human expert annotators. The evaluation follows a two-phase design: an **experimentation phase**, in which a set of commits is annotated by multiple LLMs under varying contextual configurations, and an **analysis phase**, in which the resulting annotations are compared against a human-established baseline using standard inter-rater agreement metrics.

The unit of analysis is an individual Git commit. The quality of LLM annotations is assessed by measuring their agreement with three independent human annotators who annotated the same set of commits using the same taxonomy and guidelines.

## 3.2 Annotation Taxonomy

Commits are classified along four independent dimensions, each scored on a five-point ordinal scale from 0 (not applicable) to 4 (strongly applicable):

- **Bug-Fixing Commit (BFC)**: The commit fixes a software fault that manifests as a system failure prior to the commit. Changes restricted to comments or code style cannot constitute a BFC, as they do not alter system behavior.
- **Bug-Preventing Commit (BPC)**: The commit prevents a potential future failure without fixing an already-known bug. This captures proactive changes that anticipate undiscovered faults, such as improving return values or adding defensive checks.
- **Perfective Commit (PRC)**: The commit improves code quality — through refactoring, optimization, readability improvements, or style changes — without fixing or preventing bugs or adding new functionality.
- **New Feature Commit (NFC)**: The commit introduces functionality or capabilities not previously present in the codebase, such as support for new hardware, new APIs, or new configuration options.

A commit may receive non-zero scores on more than one dimension if independent reasons justify each label. However, when a single underlying reason simultaneously satisfies multiple categories, only the highest-priority category receives a non-zero score. The enforced priority ordering is: BFC > BPC > PRC > NFC.

To avoid introducing a classification bias toward BFC, the "Fixes:" tag present in some Linux kernel commit messages — a conventional marker that explicitly references a previously reported bug — was removed from all commit messages before annotation, for both human annotators and LLM prompts.

## 3.3 Dataset

The source data consists of commits from the Linux Kernel Git repository. In prior work, a corpus of 1,000 Linux kernel commits was assembled and manually annotated by three independent human experts (referred to as annotators A, B, and C), forming the basis of the human reference dataset.

For this study, a validation subset of 50 commits was randomly sampled from the larger corpus, ensuring that all three human annotators had independently provided scores for each selected commit. This shared subset serves as the basis for all inter-rater agreement comparisons between LLMs and human annotators. Two variants of the validation set were prepared: one containing only the commit message and metadata, and one enriched with the full unified diff, change statistics, and the list of modified files, used in context configurations that include diff information (see Section 3.5.2).

## 3.4 Human Annotation Baseline

The three human annotators (A, B, C) independently assigned scores across all four dimensions for each commit in the validation subset, following the same taxonomy and contextual guidelines as those provided to the LLMs. The inter-annotator agreement among the three human annotators was computed using the same metrics applied to LLM annotations (described in Section 3.6), serving as a ceiling reference against which LLM performance is compared. A positive difference between an LLM metric and the corresponding human baseline indicates that the LLM matches or exceeds human-level agreement.

## 3.5 LLM Annotation

### 3.5.1 Models

Eleven LLMs were evaluated, covering a range of architectures, parameter scales, and inference providers:

| Model | Provider / Backend |
|---|---|
| gpt-5-mini | GitHub Copilot API |
| codellama:34b | Ollama (local) |
| codellama:70b | Ollama (local) |
| deepseek-coder:33b | Ollama (local) |
| deepseek-r1:32b | Ollama (local) |
| deepseek-r1:70b | Ollama (local) |
| gpt-oss:20b | Ollama (local) |
| gpt-oss:120b | Ollama (local) |
| llama4:16×17b | Ollama (local) |
| qwen3-coder:30b | Ollama (local) |
| gemma4:31b | Ollama (local) |

All models were run with a sampling temperature of 0.0 to maximize output determinism. The maximum response token limit was set to 10,000 to reduce the incidence of truncated JSON outputs in more verbose models.

### 3.5.2 Context Configurations

Each model was evaluated under five distinct context configurations, referred to as rounds (r1–r5), designed to progressively increase the amount and specificity of information available to the model:

| Round | Configuration | Description |
|---|---|---|
| r1 | message | Commit message only |
| r2 | message + single-label | Commit message with a single-label scoring policy |
| r3 | message + single-label + few-shot | As r2, plus human-annotated examples |
| r4 | diff + single-label | Commit message, full diff, and single-label policy |
| r5 | diff + single-label + few-shot | All of the above combined |

The **single-label** policy instructs the model to assign a high score (3 or 4) to at most one dimension per commit, unless clearly independent reasons justify multiple labels, mirroring the priority rule described in Section 3.2. The **diff** configuration supplements the commit message with the full unified diff, a summary of added and removed line counts, and the list of modified files.

### 3.5.3 Prompt Design

All prompts follow a structured template composed of the following layers, assembled in order:

1. **Annotator context**: A description of the annotation task, its purpose, and the general guidelines provided to human annotators, including the priority ordering rule for overlapping categories.
2. **Taxonomy definitions**: The formal definitions of BFC, BPC, PRC, and NFC, identical to those used in the human annotation process.
3. **Scoring rubric**: Instructions for assigning a score from 0 to 4 to each dimension, together with a self-assessed understanding score (0 = no comprehension, 4 = full comprehension of the commit's intent).
4. **Single-label policy section** *(rounds r2–r5)*: An explicit instruction to restrict high scores to a single primary category unless independent reasons justify labeling multiple dimensions.
5. **Few-shot examples section** *(rounds r3 and r5)*: Three human-annotated commit examples covering representative agreement patterns — a case of full inter-annotator agreement, a case of partial disagreement, and a case of high disagreement — selected to illustrate the breadth of annotation difficulty.
6. **Output budget constraints**: Word-count limits per output field to reduce response verbosity and the incidence of malformed or truncated JSON responses.
7. **Commit context**: The commit message with "Fixes:" tags removed, and — when diff mode is active — the unified diff, change statistics, and modified file list.
8. **Output format specification**: A requirement to respond with a structured JSON object containing the fields `understanding`, `bfc`, `bpc`, `prc`, `nfc`, and `summary`, each with a numeric score and a brief rationale.

### 3.5.4 Annotation Process and Implementation

The annotation process was orchestrated by a batch annotation script that reads the validation set, dispatches each commit to the configured LLM, and persists the result as a structured JSON file. To improve throughput, annotations were executed in parallel using a thread pool. Transient rate-limit errors were handled automatically through a retry mechanism with a configurable delay and a maximum of three attempts per commit. Commits for which an annotation file already existed were skipped, enabling incremental resumption of interrupted runs. After annotation, a post-processing step consolidated the per-commit JSON files into a single CSV file per model and round, extracting the numeric scores for each dimension along with the commit identifier.

## 3.6 Agreement Metrics

Agreement between LLM and human annotations is quantified using three complementary metrics, each assessing a different aspect of annotation quality. All metrics are computed independently for each of the four classification dimensions and then averaged to yield an overall score per model and round. For each LLM, metrics are computed on the subset of commits for which annotations from all three human annotators and the LLM are available.

### 3.6.1 Krippendorff's Alpha

Krippendorff's Alpha ($kA$) measures multi-rater agreement at the ordinal level of measurement, penalizing larger disagreements more severely than smaller ones. For the human baseline, $kA$ is computed directly on the three-annotator matrix over the shared commit subset. For each LLM, $kA$ is computed as the mean across three leave-one-out replacement permutations, in which each human annotator is substituted in turn by the LLM:

$$kA_{\text{Mean}} = \frac{1}{3}\left[kA(\text{LLM}, B, C) + kA(A, \text{LLM}, C) + kA(A, B, \text{LLM})\right]$$

The reported difference is $\Delta kA = kA_{\text{Mean}} - kA(A, B, C)$. A positive $\Delta kA$ indicates that substituting a human annotator with the LLM preserves or improves group-level agreement.

### 3.6.2 Cohen's Kappa

Cohen's Kappa ($cK$) measures pairwise agreement between two raters, using quadratic weighting to reflect the ordinal nature of the scale. For the human baseline, $cK$ is computed as the mean of the three human–human pairwise kappas:

$$cK_{\text{baseline}} = \frac{1}{3}\left[cK(A, B) + cK(A, C) + cK(B, C)\right]$$

For each LLM, $cK$ is the mean of its pairwise agreements with each human annotator:

$$cK_{\text{Mean}} = \frac{1}{3}\left[cK(A, \text{LLM}) + cK(B, \text{LLM}) + cK(C, \text{LLM})\right]$$

The reported difference is $\Delta cK = cK_{\text{Mean}} - cK_{\text{baseline}}$.

### 3.6.3 Alt-Test

The Alt-Test ($aT$) assesses the probability that an LLM annotator is closer to the human consensus than a held-out human annotator, providing a direct measure of substitutability. For each commit and each held-out human annotator $h_i$, the advantage probability is defined as:

$$\rho_f = P\!\left(\left|\text{LLM} - \hat{h}\right| < \left|h_i - \hat{h}\right|\right) + \frac{1}{2}\,P\!\left(\left|\text{LLM} - \hat{h}\right| = \left|h_i - \hat{h}\right|\right)$$

where $\hat{h}$ denotes the mean score of the remaining two human annotators. The human baseline for $aT$ is established through three leave-one-out rotations in which each human in turn acts as the focal annotator against the consensus of the other two.

The Alt-Test formalizes the null hypothesis that the LLM's expected alignment advantage $E[d]$ does not fall below a cost-benefit threshold $\varepsilon = 0.15$. A one-sided bootstrap hypothesis test with 10,000 resamples is applied, and multiple comparisons across the three per-human tests are corrected using the Benjamini–Yekutieli False Discovery Rate (FDR) procedure. Two summary statistics are reported per LLM and round:

- **$aT$ Mean** ($\rho_f$): the mean advantage probability of the LLM across all three leave-one-out perspectives, averaged over all classification dimensions.
- **Winning Rate (WR)**: the fraction of per-human hypothesis tests in which the null is rejected after FDR correction, averaged over the four classification dimensions. A WR $\geq$ 0.5 is interpreted as evidence that the LLM constitutes a viable statistical replacement for a human annotator.

## 3.7 Analysis Variants

All three metrics are computed under two analysis variants. The **primary analysis** considers the four original dimensions (BFC, BPC, PRC, NFC). The **merged-label analysis** collapses BFC and BPC into a single dimension:

$$\text{BFC}_{\text{merged}} = \max\!\left(\text{BFC}_{\text{original}},\, \text{BPC}_{\text{original}}\right)$$

reducing the label space to three dimensions: $\text{BFC}_{\text{merged}}$, PRC, and NFC. The rationale is that BFC and BPC both capture bug-related activity from complementary perspectives — reactive (fixing an existing fault) and proactive (preventing a future fault) — and merging them enables assessment of whether their combined signal aligns more consistently with human judgment. All three metrics are recomputed on this reduced label space and reported alongside the original four-label results.

## 3.8 Reproducibility

All experimental configurations are fully parameterized and reproducible. Each annotation run is specified by the model identifier, the context configuration, and the input dataset. Annotation results are persisted per model and round in structured directories. The complete analysis — including metric computation, statistical tests, and visualization — is implemented in a self-contained Jupyter notebook publicly available in the study repository.