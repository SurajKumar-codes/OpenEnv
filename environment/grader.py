"""Stateless grader for the OpenEnv Data Cleaning environment.

The ``TaskGrader`` receives everything it needs via method arguments and
returns a ``RewardInfo`` with per-step and cumulative reward breakdowns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from environment.models import Action, RewardInfo, State


class TaskGrader:
    """Scores a single agent action against the ground truth.

    The grader is **stateless per call** — all context (current DataFrame,
    ground truth, cumulative state) is passed in explicitly.

    Reward schedule
    ───────────────
    +0.10  per correctly filled missing value
    +0.10  per correctly fixed type
    +0.15  per duplicate correctly removed
    +0.20  per outlier correctly handled
    -0.10  for each correct value overwritten with wrong value
    -0.05  for each unnecessary action (no change made)
    -0.20  for dropping a row that should not be dropped
    +0.20  final bonus if submitted result matches ground truth >= 95 %
    """

    # ── reward constants ────────────────────────────────────────────
    REWARD_FILL_MISSING = 0.10
    REWARD_FIX_TYPE = 0.10
    REWARD_REMOVE_DUPLICATE = 0.15
    REWARD_HANDLE_OUTLIER = 0.20
    PENALTY_WRONG_OVERWRITE = -0.10
    PENALTY_UNNECESSARY = -0.05
    PENALTY_BAD_DROP = -0.20
    BONUS_FINAL = 0.20

    def grade(
        self,
        action: Action,
        current_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        state: State,
    ) -> RewardInfo:
        """Compute the incremental reward for *action* applied to *current_df*.

        Args:
            action: The agent's cleaning action.
            current_df: The DataFrame **after** the action has been applied.
            ground_truth_df: The gold-standard clean DataFrame.
            state: The current environment state (before this reward is added).

        Returns:
            RewardInfo: Detailed breakdown of the reward for this step.
        """
        step_reward = 0.0
        breakdown: dict[str, float] = {}
        feedback_parts: list[str] = []

        if action.action_type == "submit":
            # Final submission — compute bonus based on overall match
            step_reward, breakdown, feedback_parts = self._grade_submission(
                current_df, ground_truth_df
            )
        elif action.action_type == "drop_row":
            step_reward, breakdown, feedback_parts = self._grade_drop_row(
                action, current_df, ground_truth_df
            )
        elif action.action_type == "remove_duplicate":
            step_reward, breakdown, feedback_parts = self._grade_remove_duplicate(
                action, current_df, ground_truth_df
            )
        elif action.action_type in ("fix_cell", "fill_missing", "fix_type"):
            step_reward, breakdown, feedback_parts = self._grade_cell_fix(
                action, current_df, ground_truth_df
            )
        else:
            feedback_parts.append(f"Unknown action type: {action.action_type}")

        cumulative = state.score_so_far + step_reward

        return RewardInfo(
            step_reward=round(step_reward, 4),
            cumulative_reward=round(cumulative, 4),
            reward_breakdown=breakdown,
            feedback=" | ".join(feedback_parts) if feedback_parts else "No feedback.",
        )

    # ── private grading helpers ─────────────────────────────────────

    def _grade_cell_fix(
        self,
        action: Action,
        current_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> tuple[float, dict[str, float], list[str]]:
        """Grade fix_cell / fill_missing / fix_type actions."""
        reward = 0.0
        breakdown: dict[str, float] = {}
        feedback: list[str] = []

        row = action.row_index
        col = action.column_name

        if row is None or col is None:
            breakdown["penalty"] = self.PENALTY_UNNECESSARY
            feedback.append("Action missing row_index or column_name — no effect.")
            return self.PENALTY_UNNECESSARY, breakdown, feedback

        # Bounds check
        if row >= len(current_df) or col not in current_df.columns:
            breakdown["penalty"] = self.PENALTY_UNNECESSARY
            feedback.append("Row/column out of bounds — no effect.")
            return self.PENALTY_UNNECESSARY, breakdown, feedback

        # Check if the new value matches ground truth
        if row < len(ground_truth_df) and col in ground_truth_df.columns:
            gt_val = ground_truth_df.at[row, col]
            new_val = action.new_value

            if self._values_match(new_val, gt_val):
                # Determine the specific reward based on action_type
                if action.action_type == "fill_missing":
                    reward = self.REWARD_FILL_MISSING
                    breakdown["fill_accuracy"] = reward
                    feedback.append(f"Correctly filled missing value at ({row}, {col}).")
                elif action.action_type == "fix_type":
                    reward = self.REWARD_FIX_TYPE
                    breakdown["fix_type"] = reward
                    feedback.append(f"Correctly fixed type at ({row}, {col}).")
                else:
                    reward = self.REWARD_FILL_MISSING  # generic fix_cell
                    breakdown["fix_accuracy"] = reward
                    feedback.append(f"Correctly fixed cell at ({row}, {col}).")
            else:
                # Agent overwrote a value with something wrong
                reward = self.PENALTY_WRONG_OVERWRITE
                breakdown["penalty"] = reward
                feedback.append(
                    f"Wrong value at ({row}, {col}): "
                    f"expected {gt_val!r}, got {new_val!r}."
                )
        else:
            reward = self.PENALTY_UNNECESSARY
            breakdown["penalty"] = reward
            feedback.append("Target cell is outside ground truth range.")

        return reward, breakdown, feedback

    def _grade_drop_row(
        self,
        action: Action,
        current_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> tuple[float, dict[str, float], list[str]]:
        """Grade a drop_row action.  Penalise if the row exists in ground truth."""
        reward = 0.0
        breakdown: dict[str, float] = {}
        feedback: list[str] = []

        row = action.row_index
        if row is None:
            breakdown["penalty"] = self.PENALTY_UNNECESSARY
            feedback.append("drop_row without row_index — no effect.")
            return self.PENALTY_UNNECESSARY, breakdown, feedback

        # A row that is **not** in the ground truth is a valid drop (duplicate)
        # We approximate: if current_df has more rows than ground_truth, dropping
        # the extra ones is correct.
        if len(current_df) > len(ground_truth_df):
            reward = self.REWARD_REMOVE_DUPLICATE
            breakdown["duplicate_removal"] = reward
            feedback.append(f"Dropped likely duplicate row {row}.")
        else:
            reward = self.PENALTY_BAD_DROP
            breakdown["penalty"] = reward
            feedback.append(f"Dropped row {row} that should NOT be dropped.")

        return reward, breakdown, feedback

    def _grade_remove_duplicate(
        self,
        action: Action,
        current_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> tuple[float, dict[str, float], list[str]]:
        """Grade a remove_duplicate action."""
        reward = 0.0
        breakdown: dict[str, float] = {}
        feedback: list[str] = []

        if len(current_df) > len(ground_truth_df):
            reward = self.REWARD_REMOVE_DUPLICATE
            breakdown["duplicate_removal"] = reward
            feedback.append("Duplicate row removed successfully.")
        else:
            reward = self.PENALTY_UNNECESSARY
            breakdown["penalty"] = reward
            feedback.append("No duplicate to remove — unnecessary action.")

        return reward, breakdown, feedback

    def _grade_submission(
        self,
        current_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> tuple[float, dict[str, float], list[str]]:
        """Grade a submit action — compare current state to ground truth."""
        reward = 0.0
        breakdown: dict[str, float] = {}
        feedback: list[str] = []

        match_ratio = self._compute_match_ratio(current_df, ground_truth_df)
        breakdown["match_ratio"] = round(match_ratio, 4)

        if match_ratio >= 0.95:
            reward = self.BONUS_FINAL
            breakdown["final_bonus"] = self.BONUS_FINAL
            feedback.append(
                f"Submission matches ground truth {match_ratio:.1%} — bonus awarded!"
            )
        else:
            feedback.append(
                f"Submission matches ground truth {match_ratio:.1%} — "
                f"below 95 % threshold for bonus."
            )

        return reward, breakdown, feedback

    # ── utilities ───────────────────────────────────────────────────

    @staticmethod
    def _values_match(a: Any, b: Any) -> bool:
        """Check whether two values are semantically equal.

        Handles NaN-safe comparisons and string↔numeric coercion.
        """
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False

        # Both NaN
        try:
            if pd.isna(a) and pd.isna(b):
                return True
        except (TypeError, ValueError):
            pass

        # Try numeric comparison (tolerance for floats)
        try:
            fa, fb = float(a), float(b)
            return bool(np.isclose(fa, fb, rtol=1e-4))
        except (TypeError, ValueError):
            pass

        # String comparison (case-insensitive, stripped)
        return str(a).strip().lower() == str(b).strip().lower()

    @staticmethod
    def _compute_match_ratio(
        current_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
    ) -> float:
        """Compute cell-level match ratio between two DataFrames.

        Returns:
            float: Fraction of cells that match, in [0.0, 1.0].
        """
        if current_df.empty or ground_truth_df.empty:
            return 0.0

        # Align shapes
        min_rows = min(len(current_df), len(ground_truth_df))
        common_cols = list(
            set(current_df.columns) & set(ground_truth_df.columns)
        )
        if not common_cols:
            return 0.0

        total_cells = min_rows * len(common_cols)
        matches = 0

        for col in common_cols:
            for i in range(min_rows):
                try:
                    val_cur = current_df.at[i, col]
                    val_gt = ground_truth_df.at[i, col]
                    if TaskGrader._values_match(val_cur, val_gt):
                        matches += 1
                except (KeyError, IndexError):
                    pass

        # Penalise extra / missing rows
        row_diff_penalty = abs(len(current_df) - len(ground_truth_df))
        effective_total = total_cells + row_diff_penalty * len(common_cols)

        return matches / effective_total if effective_total > 0 else 0.0
