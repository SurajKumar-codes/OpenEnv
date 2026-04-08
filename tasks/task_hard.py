"""Hard Task: Full cleaning pipeline on a 100-row customer transaction CSV.

The agent must handle missing values, type mismatches, duplicates, statistical
outliers, and inconsistent categorical values — all in one episode.
"""

from __future__ import annotations

from tasks.base_task import BaseTask


class HardTask(BaseTask):
    """Hard difficulty — full data-cleaning pipeline across 5 error categories.

    Attributes:
        task_id: ``"hard_full_pipeline"``
        difficulty: ``"hard"``
        max_steps: 60
    """

    task_id: str = "hard_full_pipeline"
    difficulty: str = "hard"
    description: str = (
        "100-row customer transaction CSV with missing values (~10 %), "
        "type mismatches, duplicate rows, 3-sigma outliers, and "
        "inconsistent categorical values (e.g. 'USA'/'US'/'U.S.A')."
    )
    max_steps: int = 60

    def get_instructions(self) -> str:
        """Return task-specific instructions for the agent.

        Returns:
            str: Multi-line instructions describing what the agent must do.
        """
        return (
            "You are given a 100-row customer transaction CSV with multiple\n"
            "categories of data quality issues:\n\n"
            "1. **Missing values (~10 %)**: Scattered across 'customer_name',\n"
            "   'email', 'amount', 'quantity', and 'rating' columns.\n"
            "2. **Type mismatches**: Some 'amount' values are currency strings\n"
            "   (e.g. '$1,234.56') instead of floats.\n"
            "3. **Duplicate rows**: 8 exact-duplicate rows that must be removed.\n"
            "4. **Outliers**: 5 extreme values in 'amount' that are > 3 standard\n"
            "   deviations from the mean.\n"
            "5. **Inconsistent categories**: The 'country' column uses variant\n"
            "   spellings (e.g. 'USA', 'US', 'U.S.A', 'United States', 'usa').\n"
            "   Standardise to the canonical form.\n\n"
            "Your goal:\n"
            "  1. Fill or fix all missing values.\n"
            "  2. Convert currency strings to numeric floats.\n"
            "  3. Remove duplicate rows.\n"
            "  4. Handle outliers (replace with a sensible value).\n"
            "  5. Standardise country names to canonical values:\n"
            "     USA, Canada, UK, Germany, France.\n"
            "  6. Submit your result when done.\n\n"
            "Recommended order: duplicates → types → missing → outliers → categories.\n\n"
            "Available actions: fix_cell, fill_missing, fix_type, drop_row,\n"
            "                   remove_duplicate, submit\n"
            "You have a maximum of 60 steps."
        )
