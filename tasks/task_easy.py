"""Easy Task: Fix missing values in a 10-row employee CSV.

The agent must identify and fill 5 missing values (NaN) across the
``salary``, ``department``, and ``age`` columns.
"""

from __future__ import annotations

from tasks.base_task import BaseTask


class EasyTask(BaseTask):
    """Easy difficulty — fill missing values in a small employee dataset.

    Attributes:
        task_id: ``"easy_missing_values"``
        difficulty: ``"easy"``
        max_steps: 10
    """

    task_id: str = "easy_missing_values"
    difficulty: str = "easy"
    description: str = (
        "10-row employee CSV with 5 missing values injected in the "
        "'salary', 'department', and 'age' columns."
    )
    max_steps: int = 10

    def get_instructions(self) -> str:
        """Return task-specific instructions for the agent.

        Returns:
            str: Multi-line instructions describing what the agent must do.
        """
        return (
            "You are given a 10-row employee CSV with 5 missing values (NaN).\n"
            "The missing values are in the 'salary', 'department', and 'age' columns.\n\n"
            "Your goal:\n"
            "  1. Identify every cell that contains a missing value.\n"
            "  2. Fill each missing value with a reasonable replacement:\n"
            "     - For 'salary': use the median salary of the dataset.\n"
            "     - For 'department': infer from context or use the most common department.\n"
            "     - For 'age': use the median age of the dataset.\n"
            "  3. Submit your result when done.\n\n"
            "Available actions: fill_missing, fix_cell, submit\n"
            "You have a maximum of 10 steps."
        )
