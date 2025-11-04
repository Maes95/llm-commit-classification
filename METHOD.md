# Methodology for Commit Annotation Using Large Language Models

## Objective

The primary objective of this experimental study is to evaluate the capability of Large Language Models (LLMs) to annotate software commits from the Linux Kernel repository across four distinct dimensions: Bug-Fixing Commits (BFC), Bug-Preventing Commits (BPC), Perfective Commits (PRC), and New Feature Commits (NFC). Unlike traditional single-label classification approaches, this research employs a multi-dimensional annotation scheme where each commit receives a score from 0 to 4 for each category, reflecting the degree to which the commit exhibits characteristics of that category. This approach acknowledges the reality that commits often serve multiple purposes simultaneously—for example, a commit might both fix a bug and improve code quality, or introduce a new feature while preventing potential future bugs. The study aims to assess whether LLMs can replicate expert human judgment in producing nuanced, multi-dimensional commit annotations that capture this complexity. We seek to determine the inter-rater reliability between LLM annotations and expert human annotations, measured through correlation coefficients and mean absolute error for each dimension. Furthermore, we aim to identify which contextual elements (commit message alone, commit message with diff, or commit message with full metadata including email discussions) contribute most significantly to annotation accuracy and consistency. The experimental results will provide insights into the viability of using LLMs for large-scale repository mining tasks requiring nuanced understanding of commit semantics, and will establish baseline performance metrics for future comparative studies in automated software evolution analysis.

## Materials

The experimental dataset consists of 1,000 commits randomly sampled from the Linux Kernel repository, stored in JSONL (JSON Lines) format in the file `data/1000-linux-commits.jsonl`. Each commit record contains comprehensive metadata including commit hash, author information, commit date, commit message, list of modified files with their change statistics (additions/deletions), parent commit references, and signed-off-by information extracted using the Perceval tool. The annotation taxonomy is defined in `documentation/definitions.md`, which provides detailed definitions for the four commit dimensions based on software engineering principles and failure/fault theory. Each dimension represents a distinct aspect of commit purpose: BFC captures bug-fixing intent, BPC captures preventive maintenance, PRC captures quality improvements without behavioral changes, and NFC captures functionality additions.

For the LLM infrastructure, we utilize the LangChain library (Python) to interface with multiple language models through OpenRouter as the API gateway provider. The models selected for evaluation include: 

- Meta's Llama 4 Maverick 
- deepseek/deepseek-chat-v3.1:free 
- meituan/longcat-flash-chat:free 
- openai/gpt-oss-20b:free 
- qwen/qwen3-coder:free 
- moonshotai/kimi-k2:free 
- google/gemma-3n-e2b-it:free 
- tngtech/deepseek-r1t2-chimera:free 
- mistralai/mistral-small-3.2-24b-instruct:free 
- google/gemini-2.0-flash-exp:free 

> I have selected some of the most relevant "free" models I have found, but I am unable to define a selection criterion that makes sense

These models represent diverse architectural approaches and training paradigms, enabling comprehensive performance comparison. The selection criteria prioritize models with proven reasoning capabilities, sufficient context window sizes (minimum 32K tokens) to accommodate complete commit information including extensive code diffs, and demonstrated proficiency in multi-aspect analysis tasks. 

## Procedure

The experimental procedure is structured as a five-phase pipeline executed sequentially to ensure systematic and reproducible commit annotation:

### Phase 1: Data Preparation

The dataset undergoes preprocessing to structure commit information at three granularity levels. For each commit in `1000-linux-commits.jsonl`, we extract:

- **Level 1 (Minimal)**: Commit message text only from the `data.message` field
- **Level 2 (Standard)**: Commit message plus file modification statistics (files added/modified/deleted, lines changed) extracted from the `data.files` array, including file paths and change counts
- **Level 3 (Full)**: Complete context including commit message, full code diff reconstructed from file changes and associated email thread discussions when available

Code diffs are reconstructed by parsing the `files` array, which contains per-file change information including `added` and `removed` line counts, file paths, and action types (Modified, Added, Deleted). For commits with email references (extracted from `Link:` tags in commit messages using regex pattern `Link: https://lore.kernel.org/r/([^\s]+)`), we fetch HTML content from the Linux Kernel Mailing List archives (lore.kernel.org) using HTTP requests, convert to plain text using BeautifulSoup, and preserve threading structure by parsing `In-Reply-To` and `References` headers.

### Phase 2: Prompt Engineering

We design a structured prompt template that combines the annotation taxonomy with commit context and implements a multi-dimensional scoring system. The template follows this architecture:

```
[SYSTEM INSTRUCTION]
You are an expert software engineering analyst specializing in commit annotation.
Your task is to evaluate commits across multiple dimensions simultaneously.

[TAXONOMY DEFINITIONS]
Below are the definitions and categories you must use for annotation:
{definitions.md}

[TASK INSTRUCTION]
Annotate the following commit by assigning a score from 0 to 4 for EACH of the four categories:
- Bug-Fixing Commit (BFC)
- Bug-Preventing Commit (BPC)
- Perfective Commit (PRC)
- New Feature Commit (NFC)

Scoring rubric:
0 = Not applicable - The commit shows no characteristics of this category
1 = Minimal - The commit shows very slight or tangential characteristics
2 = Moderate - The commit partially exhibits characteristics of this category
3 = Strong - The commit clearly exhibits characteristics of this category
4 = Primary - This is a primary or dominant characteristic of the commit

Use chain-of-thought reasoning for each dimension:
1. Identify evidence in the commit relevant to this dimension
2. Evaluate the strength and significance of this evidence
3. Assign an appropriate score based on the rubric

[COMMIT CONTEXT]
{context_at_specified_granularity_level}

[OUTPUT FORMAT]
Provide your annotation as a valid JSON object with the following structure:
{
  "bfc": {
    "score": [0-4],
    "reasoning": "Detailed explanation of BFC score"
  },
  "bpc": {
    "score": [0-4],
    "reasoning": "Detailed explanation of BPC score"
  },
  "prc": {
    "score": [0-4],
    "reasoning": "Detailed explanation of PRC score"
  },
  "nfc": {
    "score": [0-4],
    "reasoning": "Detailed explanation of NFC score"
  },
  "summary": "Brief synthesis of the commit's primary purposes"
}

Ensure your response is ONLY the JSON object, with no additional text before or after.
```

For each context level, the `{context_at_specified_granularity_level}` placeholder is populated with the corresponding preprocessed data. 

### Phase 3: LLM Configuration

Each model is configured with deterministic parameters to maximize reproducibility (if supported by the model):

```python
llm_config = {
    "temperature": 0.0,        # Eliminate randomness in token selection
    "top_p": 1.0,              # Consider full probability distribution
    "max_tokens": 3072,        # Allow detailed reasoning per dimension
    "frequency_penalty": 0.0,  # No penalty for token repetition
    "presence_penalty": 0.0,   # No penalty for topic repetition
}
```

### Phase 4: Prompt Validation and Refinement

Before annotating the full dataset, we implement an **Iterative Refinement Process** [1] to validate and optimize the prompt configuration. This methodology uses a validation subset to ensure high-quality annotations before full-scale deployment:

**Step 1 - Validation Sample Selection**: Select a stratified validation sample of 50 commits from the full dataset, ensuring representation across different Linux Kernel subsystems (drivers, core kernel, filesystems, networking) and preliminary commit types. These commits are set aside exclusively for prompt refinement and are not used in the final evaluation to avoid data leakage.

**Step 2 - Initial LLM Annotation**: Apply the initial prompt (defined in Phase 2) to annotate the validation sample using each selected LLM model. Compare these LLM annotations against the existing human ground truth annotations for these 50 commits.

**Step 3 - Statistical Validation**: Calculate agreement coefficients, specifically Cohen's kappa (per dimension), between LLM and human annotations on the validation sample. This quantitatively assesses the alignment between LLM outputs and expected results, providing an objective measure of annotation quality.

**Step 4 - Criteria Assessment**: Evaluate whether the agreement coefficient meets a predefined criterion (Cohen's kappa > 0.8 per dimension, indicating strong agreement). If the criterion is met for all dimensions, proceed to Phase 5 with the validated prompt. If not, continue to Step 5 for prompt refinement.

**Step 5 - Error Analysis and Prompt Refinement**: When agreement falls below the threshold, perform detailed error analysis to identify systematic issues. Examine commits where LLM and human annotations diverge significantly (difference ≥2 points) to understand failure patterns. Systematically refine the prompt based on identified issues:

- **Simplifying language**: Use clearer, more straightforward language to improve LLM comprehension
- **Clarifying instructions**: Rewrite ambiguous sections to eliminate multiple interpretations
- **Adding few-shot examples**: Include 2-3 concrete examples demonstrating correct annotation reasoning for each dimension
- **Adjusting context emphasis**: Modify instructions to better balance commit message, code changes, and metadata
- **Refining scoring rubric**: Add specific criteria or boundary cases to clarify score distinctions (e.g., when to assign 2 vs. 3)

**Step 6 - Iterative Repetition**: Return to Step 2 with the refined prompt and re-annotate the validation sample. Continue this cycle until achieving the target agreement level or reaching a maximum of 5 iterations. Document each iteration: prompt version, per-dimension kappa scores, specific modifications made, and rationale for changes.

Once the prompt achieves satisfactory agreement on the validation sample, the validated prompt configuration is finalized for full-scale deployment.

### Phase 5: Full Dataset Annotation

With the validated prompt from Phase 4, we proceed to annotate all 1,000 commits across all model-context combinations:

**Annotation Execution**: For each combination of model (10 models), context level (3 levels: minimal, standard, full), and commit (1,000 commits), we execute the LLM annotation process. This generates 30,000 total annotations (10 models × 3 context levels × 1,000 commits).

**Quality Assurance**: After completion, perform sanity checks on the annotation dataset:
- Verify all commits have annotations for all model-context combinations
- Check JSON parsing success rate (target: >99%)
- Validate score ranges (all scores 0-4)
- Identify and flag any anomalous patterns (e.g., identical annotations across many commits suggesting model errors)

The validation sample (50 commits from Phase 4) is excluded from all subsequent analyses to maintain methodological rigor and prevent overfitting to the validation set.

### Phase 6: Evaluation Against Ground Truth

The full dataset of 1,000 commits (excluding the 50-commit validation sample) has been previously annotated by expert software engineers, establishing a gold standard ground truth dataset of 950 commits. Each commit has been manually scored across all four dimensions (BFC, BPC, PRC, NFC) using the same 0-4 rubric provided to the LLMs.

**Inter-Annotator Agreement Verification**: Before using the human annotations as ground truth, we verify their reliability by calculating inter-annotator agreement metrics.

> This work is partially done but not finished yet. We have three files with annotations for each annotator. The first was an individual annotation, in the second there was a discussion of the commits where there were disagreements, after which the annotation could be modified. In the third, a fourth person (Jesús) consulted the annotation to try to reach a decision. After these annotations, there were still disagreements in ~20 commits. The problem is that even if we agreed on the classification of the commits, it is not exactly the same for each annotator, since although an agreement was reached on some (for example, BFC=4), other values may not coincide (we all put BFC=3 but a different value in BPC=[0-2]).

**Evaluation Metrics Computation**: LLM annotations are systematically compared against ground truth across all model-context combinations on the 950-commit evaluation set. For each dimension, we compute:

1. **Binary classification metrics** at two thresholds (strict: score ≥3; permissive: score ≥2): precision, recall, F1-score, and accuracy
2. **Score-level agreement**: Cohen's kappa (unweighted and quadratic-weighted), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE)
3. **Correlation measures**: Pearson and Spearman correlations between LLM and ground truth scores
4. **Percentage metrics**: Exact matches and within-1-point agreement rates

**Statistical Significance Testing**: To compare model performance, we apply appropriate hypothesis tests: McNemar's test for binary classification comparisons, Wilcoxon signed-rank test for score-level comparisons between model pairs, and Friedman test with post-hoc Nemenyi tests for simultaneous comparison of all models. All tests use α = 0.05 with Bonferroni correction for multiple comparisons.

## Data Analysis

The analysis phase employs comprehensive quantitative and qualitative methodologies to interpret evaluation results and extract meaningful insights about LLM performance:

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

**Cost-Effectiveness Optimization**: We analyze operational trade-offs:
- **Token consumption**: Total and per-commit token usage across model-context combinations
- **API cost estimation**: Calculate monetary costs based on November 2025 provider pricing
- **Accuracy-cost frontier**: Identify Pareto-optimal configurations balancing performance and expense

### Qualitative Analysis

**Reasoning Quality Assessment**: Using a sample of commits annotated by LLMs, we can evaluate:
- **Evidence grounding**: Whether reasoning explicitly cites commit message text, code changes, or metadata
- **Definition alignment**: Correct application of category definitions from the taxonomy
- **Logical coherence**: Internal consistency and sound argumentation structure
- **Score justification**: Clear connection between presented evidence and assigned score

## Bibliography

[1] V. De Martino, J. Castaño, F. Palomba, X. Franch and S. Martínez-Fernández, "A Framework for Using LLMs for Repository Mining Studies in Empirical Software Engineering," 2025 IEEE/ACM International Workshop on Methodological Issues with Empirical Studies in Software Engineering (WSESE), Ottawa, ON, Canada, 2025, pp. 6-11, doi: 10.1109/WSESE66602.2025.00008.