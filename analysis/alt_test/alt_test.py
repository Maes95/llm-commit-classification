"""Utilities for Alternative Annotator testing.

This module extracts the statistical logic from the notebook so it can be reused
from scripts and tests.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import t as t_dist
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


def compute_rmse_score(f_annotation: float, other_annotations: np.ndarray) -> float:
    """Compute negative RMSE alignment score (higher is better)."""
    return -float(np.sqrt(np.mean((f_annotation - other_annotations) ** 2)))


def compute_advantage_probabilities(
    df,
    llm_name: str,
    human_annotators: list[str],
    score_func: Callable[[float, np.ndarray], float] = compute_rmse_score,
) -> dict[str, dict[str, object]]:
    """Compute advantage probabilities for one LLM against each excluded human."""
    results: dict[str, dict[str, object]] = {}

    for excluded_human in human_annotators:
        remaining_humans = [h for h in human_annotators if h != excluded_human]
        w_f_list: list[int] = []
        w_h_list: list[int] = []

        for _, row in df.iterrows():
            llm_annotation = row[llm_name]
            excluded_annotation = row[excluded_human]
            remaining_annotations = row[remaining_humans].values

            s_llm = score_func(llm_annotation, remaining_annotations)
            s_human = score_func(excluded_annotation, remaining_annotations)

            w_f = 1 if s_llm >= s_human else 0
            w_h = 1 if s_human >= s_llm else 0

            w_f_list.append(w_f)
            w_h_list.append(w_h)

        results[excluded_human] = {
            "rho_f": float(np.mean(w_f_list)),
            "rho_h": float(np.mean(w_h_list)),
            "W_f": w_f_list,
            "W_h": w_h_list,
            "n_instances": len(w_f_list),
        }

    return results


def bootstrap_hypothesis_test(
    d: np.ndarray,
    epsilon: float,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, float | str]:
    """Perform one-sided bootstrap test for H1: E[d] < epsilon."""
    n = len(d)
    rng = np.random.default_rng(seed=seed)
    bootstrap_means = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = rng.choice(d, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    p_value = float(np.mean(bootstrap_means >= epsilon))
    test_stat = float(np.mean(d))

    return {
        "p_value": p_value,
        "test_stat": test_stat,
        "ci_lower": float(np.percentile(bootstrap_means, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_means, 97.5)),
        "test_name": "bootstrap",
    }


def ttest_hypothesis_test(
    d: np.ndarray,
    epsilon: float,
) -> dict[str, float | str]:
    """Perform one-sample t-test for H1: E[d] < epsilon."""
    n = len(d)
    t_stat, p_value = ttest_1samp(d, epsilon, alternative="less")

    mean_d = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    t_critical = float(t_dist.ppf(0.975, df=n - 1))

    return {
        "p_value": float(p_value),
        "test_stat": float(t_stat),
        "ci_lower": mean_d - t_critical * se,
        "ci_upper": mean_d + t_critical * se,
        "test_name": "t_test",
    }


def benjamini_yekutieli_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Apply Benjamini-Yekutieli FDR correction."""
    rejected, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_by")
    return rejected


def perform_alt_test(
    advantage_results: dict[str, dict[str, object]],
    epsilon: float = 0.0,
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    hypothesis_test: str = "t_test",
) -> dict[str, object]:
    """Run the alternative annotator test with BY correction.

    Args:
        advantage_results: Output of compute_advantage_probabilities.
        epsilon: Cost-benefit threshold epsilon.
        alpha: Significance level.
        n_bootstrap: Number of bootstrap samples if bootstrap test is used.
        hypothesis_test: One of {"t_test", "bootstrap"}.
    """
    test_results: dict[str, dict[str, object]] = {}
    p_values_list: list[float] = []
    human_list: list[str] = []

    for human, res in advantage_results.items():
        d = np.array(res["W_h"]) - np.array(res["W_f"])
        n = len(d)

        rho_f = float(res["rho_f"])
        rho_h = float(res["rho_h"])

        if hypothesis_test == "bootstrap":
            test_result = bootstrap_hypothesis_test(d, epsilon, n_bootstrap=n_bootstrap)
        elif hypothesis_test == "t_test":
            test_result = ttest_hypothesis_test(d, epsilon)
        else:
            raise ValueError("hypothesis_test must be 't_test' or 'bootstrap'")

        d_bar = float(np.mean(d))
        s = float(np.std(d, ddof=1))

        test_results[human] = {
            "rho_f": rho_f,
            "rho_h": rho_h,
            "rho_diff": rho_f - rho_h,
            "d_bar": d_bar,
            "s": s,
            "test_used": test_result["test_name"],
            "test_stat": test_result["test_stat"],
            "p_value": test_result["p_value"],
            "n": n,
            "bootstrap_ci_lower": test_result["ci_lower"],
            "bootstrap_ci_upper": test_result["ci_upper"],
        }

        p_values_list.append(float(test_result["p_value"]))
        human_list.append(human)

    p_values_array = np.array(p_values_list)
    rejected = benjamini_yekutieli_correction(p_values_array, alpha)

    for i, human in enumerate(human_list):
        test_results[human]["rejected"] = bool(rejected[i])

    winning_rate = float(np.mean(rejected))
    avg_advantage_prob = float(np.mean([res["rho_f"] for res in test_results.values()]))

    return {
        "test_results": test_results,
        "winning_rate": winning_rate,
        "avg_advantage_prob": avg_advantage_prob,
        "can_replace": winning_rate >= 0.5,
    }
