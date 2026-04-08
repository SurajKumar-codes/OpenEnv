"""Medium Task: Fix types, duplicates, and negative values in a 30-row sales CSV.

The agent must standardise prices from currency strings, normalise mixed date
formats, remove exact duplicate rows, and correct negative quantities.
"""

from __future__ import annotations

from tasks.base_task import BaseTask


class MediumTask(BaseTask):
    """Medium difficulty — fix types, dates, duplicates, and negative quantities.

    Attributes:
        task_id: ``"medium_type_and_duplicates"``
        difficulty: ``"medium"``
        max_steps: 25
    """

    task_id: str = "medium_type_and_duplicates"
    difficulty: str = "medium"
    description: str = (
        "30-row sales CSV with price stored as currency strings, mixed date "
        "formats, 5 exact duplicate rows, and negative quantity values."
    )
    max_steps: int = 25

    def get_instructions(self) -> str:
        """Return task-specific instructions for the agent.

        Returns:
            str: Multi-line instructions describing what the agent must do.
        """
        return (
            "You are given a 30-row sales CSV containing four categories of errors:\n\n"
            "1. **Type errors in 'price'**: Values are stored as currency strings\n"
            "   (e.g. '$1,200.00') instead of plain floats.\n"
            "2. **Mixed date formats in 'date'**: Dates use different formats such as\n"
            "   MM/DD/YYYY, DD-Mon-YYYY, YYYY.MM.DD, etc. Standardise to ISO 8601\n"
            "   (YYYY-MM-DD).\n"
            "3. **Duplicate rows**: There are 5 exact-duplicate rows that must be\n"
            "   removed.\n"
            "4. **Negative quantities**: Some 'quantity' values are negative and\n"
            "   should be corrected to their absolute value.\n\n"
            "Your goal:\n"
            "  1. Fix all price values to plain floats.\n"
            "  2. Standardise all dates to YYYY-MM-DD.\n"
            "  3. Remove all exact duplicate rows.\n"
            "  4. Fix negative quantities.\n"
            "  5. Submit your result when done.\n\n"
            "Available actions: fix_type, fix_cell, remove_duplicate, drop_row, submit\n"
            "You have a maximum of 25 steps."
        )
