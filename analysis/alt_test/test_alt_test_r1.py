"""Integration test for alt_test on GPT-OSS-20B round 1 data."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from analysis.alt_test.alt_test import compute_advantage_probabilities, perform_alt_test


REPO_ROOT = Path(__file__).resolve().parents[2]
HUMAN_DATA_DIR = REPO_ROOT / "data" / "human-annotator-results"
LLM_R1_FILE = REPO_ROOT / "data" / "llm-annotator-results" / "r1" / "annotations_ollama_gpt-oss_20b.csv"
LLM_R2_FILE = REPO_ROOT / "data" / "llm-annotator-results" / "r2" / "annotations_ollama_gpt-oss_20b.csv"
HUMAN_ANNOTATORS = ["A", "B", "C"]
LLM_NAME_R1 = "gpt-oss_20b_r1"
LLM_NAME_R2 = "gpt-oss_20b_r2"
EPSILON = 0.15
ALPHA = 0.05
CATEGORIES = ["bfc", "bpc", "prc", "nfc"]


def build_unified_category_df(category: str, llm_file: Path, llm_name: str) -> pd.DataFrame:
    """Build category dataframe with columns: hash, A, B, C, llm_name."""
    human_data = {
        annotator: pd.read_csv(HUMAN_DATA_DIR / f"annotations_{annotator}.csv")
        for annotator in HUMAN_ANNOTATORS
    }
    llm_df = pd.read_csv(llm_file)

    category_df = human_data["A"][["hash", category]].rename(columns={category: "A"}).copy()

    for annotator in ["B", "C"]:
        category_df = category_df.merge(
            human_data[annotator][["hash", category]].rename(columns={category: annotator}),
            on="hash",
            how="inner",
        )

    category_df = category_df.merge(
        llm_df[["hash", category]].rename(columns={category: llm_name}),
        on="hash",
        how="inner",
    )

    return category_df.dropna()


def build_unified_general_df(llm_file: Path, llm_name: str) -> pd.DataFrame:
    """Build one global dataframe by stacking all categories."""
    frames = [build_unified_category_df(category, llm_file, llm_name) for category in CATEGORIES]
    return pd.concat(frames, ignore_index=True)


class TestAltTestR1(unittest.TestCase):
    def test_gpt_oss_20b_round1_general_result(self) -> None:
        expected = {
            "winning_rate": 0.0,
            "avg_advantage_prob": 0.8316666666666667,
            "can_replace": False,
            "rejected": {"A": False, "B": False, "C": False},
        }

        df = build_unified_general_df(LLM_R1_FILE, LLM_NAME_R1)
        adv_probs = compute_advantage_probabilities(df, LLM_NAME_R1, HUMAN_ANNOTATORS)
        result = perform_alt_test(adv_probs, epsilon=EPSILON, alpha=ALPHA)

        self.assertAlmostEqual(result["winning_rate"], expected["winning_rate"], places=12)
        self.assertAlmostEqual(
            result["avg_advantage_prob"],
            expected["avg_advantage_prob"],
            places=12,
        )
        self.assertEqual(result["can_replace"], expected["can_replace"])

        for human in HUMAN_ANNOTATORS:
            self.assertEqual(
                result["test_results"][human]["rejected"],
                expected["rejected"][human],
            )

    def test_gpt_oss_20b_round2_general_result(self) -> None:
        expected = {
            "winning_rate": 1.0,
            "avg_advantage_prob": 0.8883333333333333,
            "can_replace": True,
            "rejected": {"A": True, "B": True, "C": True},
        }

        df = build_unified_general_df(LLM_R2_FILE, LLM_NAME_R2)
        adv_probs = compute_advantage_probabilities(df, LLM_NAME_R2, HUMAN_ANNOTATORS)
        result = perform_alt_test(adv_probs, epsilon=EPSILON, alpha=ALPHA)

        self.assertAlmostEqual(result["winning_rate"], expected["winning_rate"], places=12)
        self.assertAlmostEqual(
            result["avg_advantage_prob"],
            expected["avg_advantage_prob"],
            places=12,
        )
        self.assertEqual(result["can_replace"], expected["can_replace"])

        for human in HUMAN_ANNOTATORS:
            self.assertEqual(
                result["test_results"][human]["rejected"],
                expected["rejected"][human],
            )


if __name__ == "__main__":
    unittest.main()
