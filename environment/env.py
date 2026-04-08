"""Core OpenEnv environment for CSV data-cleaning evaluation.

Usage::

    from environment.env import DataCleaningEnv

    env = DataCleaningEnv()
    obs = env.reset("easy_missing_values")
    obs, reward, done, info = env.step(action)
    state = env.state()
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from environment.grader import TaskGrader
from environment.models import Action, Observation, RewardInfo, State
from tasks.task_easy import EasyTask
from tasks.task_medium import MediumTask
from tasks.task_hard import HardTask


# ── task registry ────────────────────────────────────────────────────
_TASK_REGISTRY: dict[str, Any] = {
    "easy_missing_values": EasyTask,
    "medium_type_and_duplicates": MediumTask,
    "hard_full_pipeline": HardTask,
}


class DataCleaningEnv:
    """OpenEnv-compliant environment for LLM agent data-cleaning evaluation.

    The environment loads a dirty CSV, lets the agent submit cleaning
    actions one at a time, and returns incremental rewards.

    Lifecycle::

        obs = env.reset(task_id)
        while not done:
            obs, reward, done, info = env.step(action)
        final_state = env.state()
    """

    def __init__(self) -> None:
        """Initialise the environment with default (empty) internal state."""
        self._task: Any | None = None
        self._current_df: pd.DataFrame | None = None
        self._ground_truth_df: pd.DataFrame | None = None
        self._original_dirty_df: pd.DataFrame | None = None
        self._grader = TaskGrader()

        # State tracking
        self._task_id: str = ""
        self._steps_taken: int = 0
        self._max_steps: int = 0
        self._total_errors: int = 0
        self._errors_fixed: int = 0
        self._errors_introduced: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False

    # ── public interface ────────────────────────────────────────────

    def reset(self, task_id: str) -> Observation:
        """Reset the environment for a new episode.

        Args:
            task_id: Identifier of the task to load (e.g. ``"easy_missing_values"``).

        Returns:
            Observation: The initial observation the agent receives.

        Raises:
            ValueError: If *task_id* is not in the registry.
        """
        if task_id not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id {task_id!r}. "
                f"Available: {list(_TASK_REGISTRY)}"
            )

        task_cls = _TASK_REGISTRY[task_id]
        self._task = task_cls()
        self._task_id = task_id

        self._ground_truth_df = self._task.get_clean_data()
        self._original_dirty_df = self._task.get_dirty_data()
        self._current_df = self._original_dirty_df.copy()

        self._max_steps = self._task.max_steps
        self._steps_taken = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._errors_fixed = 0
        self._errors_introduced = 0

        # Count initial errors
        self._total_errors = self._count_errors(
            self._original_dirty_df, self._ground_truth_df
        )

        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """Apply an agent action and return the result.

        Args:
            action: The cleaning action the agent wants to perform.

        Returns:
            tuple: ``(observation, reward, done, info)``

        Raises:
            RuntimeError: If the environment has not been reset or the episode
                is already done.
        """
        if self._current_df is None or self._task is None:
            raise RuntimeError("Call env.reset(task_id) before env.step().")
        if self._done:
            raise RuntimeError("Episode is done. Call env.reset() to start a new one.")

        # Apply the action to the working DataFrame
        self._apply_action(action)

        self._steps_taken += 1

        # Grade the action
        current_state = self._build_state()
        reward_info: RewardInfo = self._grader.grade(
            action=action,
            current_df=self._current_df,
            ground_truth_df=self._ground_truth_df,
            state=current_state,
        )

        self._cumulative_reward = reward_info.cumulative_reward

        # Track error changes
        if reward_info.step_reward > 0 and action.action_type != "submit":
            self._errors_fixed += 1
        elif reward_info.step_reward < 0 and "penalty" in reward_info.reward_breakdown:
            if reward_info.reward_breakdown.get("penalty", 0) == TaskGrader.PENALTY_WRONG_OVERWRITE:
                self._errors_introduced += 1

        # Check termination conditions
        if action.action_type == "submit":
            self._done = True
        elif self._steps_taken >= self._max_steps:
            self._done = True

        obs = self._make_observation()
        info = {
            "reward_info": reward_info.model_dump(),
            "steps_remaining": self._max_steps - self._steps_taken,
        }

        # Normalise the cumulative reward to [0, 1]
        max_possible = self._total_errors * 0.15 + self.BONUS_FINAL  # rough upper bound
        normalised_reward = max(0.0, min(1.0, self._cumulative_reward / max_possible)) if max_possible > 0 else 0.0

        return obs, round(normalised_reward, 4), self._done, info

    # bonus constant shortcut
    BONUS_FINAL = TaskGrader.BONUS_FINAL

    def state(self) -> State:
        """Return the current internal state of the environment.

        Returns:
            State: A snapshot of steps taken, errors fixed, score, etc.
        """
        return self._build_state()

    # ── private helpers ─────────────────────────────────────────────

    def _apply_action(self, action: Action) -> None:
        """Mutate ``self._current_df`` according to the given action."""
        assert self._current_df is not None

        if action.action_type == "submit":
            return  # No mutation — just triggers end-of-episode grading.

        if action.action_type == "drop_row":
            if action.row_index is not None and 0 <= action.row_index < len(self._current_df):
                self._current_df = self._current_df.drop(
                    index=action.row_index
                ).reset_index(drop=True)
            return

        if action.action_type == "remove_duplicate":
            if action.row_index is not None and 0 <= action.row_index < len(self._current_df):
                self._current_df = self._current_df.drop(
                    index=action.row_index
                ).reset_index(drop=True)
            else:
                # Remove first exact duplicate found
                self._current_df = self._current_df.drop_duplicates(keep="first").reset_index(drop=True)
            return

        # Cell-level actions: fix_cell, fill_missing, fix_type
        if action.row_index is not None and action.column_name is not None:
            row, col = action.row_index, action.column_name
            if 0 <= row < len(self._current_df) and col in self._current_df.columns:
                self._current_df.at[row, col] = action.new_value

    def _make_observation(self) -> Observation:
        """Build an ``Observation`` from the current environment state."""
        assert self._current_df is not None and self._task is not None

        errors = self._detect_errors(self._current_df, self._ground_truth_df)

        return Observation(
            task_id=self._task_id,
            instructions=self._task.get_instructions(),
            data_snapshot=self._current_df.to_csv(index=False),
            errors_detected=errors,
            current_step=self._steps_taken,
            max_steps=self._max_steps,
            done=self._done,
        )

    def _build_state(self) -> State:
        """Construct the State object."""
        max_possible = self._total_errors * 0.15 + self.BONUS_FINAL
        score = max(0.0, min(1.0, self._cumulative_reward / max_possible)) if max_possible > 0 else 0.0

        return State(
            task_id=self._task_id,
            steps_taken=self._steps_taken,
            total_errors=self._total_errors,
            errors_fixed=self._errors_fixed,
            errors_introduced=self._errors_introduced,
            score_so_far=round(score, 4),
            is_complete=self._done,
        )

    # ── error detection / counting ──────────────────────────────────

    @staticmethod
    def _count_errors(dirty_df: pd.DataFrame, clean_df: pd.DataFrame) -> int:
        """Count the total number of cell-level discrepancies + extra rows."""
        count = 0
        common_cols = list(set(dirty_df.columns) & set(clean_df.columns))
        min_rows = min(len(dirty_df), len(clean_df))

        for col in common_cols:
            for i in range(min_rows):
                try:
                    dv = dirty_df.at[i, col]
                    cv = clean_df.at[i, col]
                    if not TaskGrader._values_match(dv, cv):
                        count += 1
                except (KeyError, IndexError):
                    pass

        # Extra rows in dirty = duplicates
        count += max(0, len(dirty_df) - len(clean_df))
        return count

    @staticmethod
    def _detect_errors(
        current_df: pd.DataFrame, ground_truth_df: pd.DataFrame | None
    ) -> list[str]:
        """Return human-readable error descriptions visible to the agent."""
        errors: list[str] = []

        # Missing values
        for col in current_df.columns:
            nulls = current_df[col].isna().sum()
            if nulls > 0:
                errors.append(f"Column '{col}' has {nulls} missing value(s).")

        # Possible duplicates
        dup_count = current_df.duplicated().sum()
        if dup_count > 0:
            errors.append(f"{dup_count} duplicate row(s) detected.")

        # Type hints: numeric columns with non-numeric strings
        for col in current_df.columns:
            if current_df[col].dtype == object:
                sample = current_df[col].dropna().head(5).tolist()
                has_dollar = any(
                    isinstance(v, str) and v.startswith("$") for v in sample
                )
                if has_dollar:
                    errors.append(
                        f"Column '{col}' may contain currency-formatted strings."
                    )

        return errors
