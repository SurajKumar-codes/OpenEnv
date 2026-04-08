"""Abstract base class for all OpenEnv data-cleaning tasks.

Every concrete task (easy, medium, hard) inherits from ``BaseTask`` and
implements the hooks that load data and provide instructions.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


# Resolve the project-level ``data/`` directory regardless of working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


class BaseTask(ABC):
    """Blueprint that every task must follow.

    Attributes:
        task_id: Unique slug for the task (e.g. ``"easy_missing_values"``).
        difficulty: One of ``easy``, ``medium``, ``hard``.
        description: Short human-readable description of the task.
        max_steps: Maximum actions the agent may take.
    """

    task_id: str
    difficulty: str
    description: str
    max_steps: int

    # ── data helpers ────────────────────────────────────────────────

    def _data_dir(self) -> Path:
        """Return the ``data/<difficulty>/`` directory for this task."""
        return _DATA_DIR / self.difficulty

    def get_dirty_data(self) -> pd.DataFrame:
        """Load and return the dirty CSV as a DataFrame.

        Returns:
            pd.DataFrame: The messy input the agent must clean.

        Raises:
            FileNotFoundError: If the dirty CSV has not been generated yet.
        """
        path = self._data_dir() / "dirty.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Dirty data not found at {path}. "
                "Run scripts/generate_datasets.py first."
            )
        return pd.read_csv(path)

    def get_clean_data(self) -> pd.DataFrame:
        """Load and return the ground-truth clean CSV as a DataFrame.

        Returns:
            pd.DataFrame: The gold-standard output the agent is scored against.

        Raises:
            FileNotFoundError: If the clean CSV has not been generated yet.
        """
        path = self._data_dir() / "clean.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Clean data not found at {path}. "
                "Run scripts/generate_datasets.py first."
            )
        return pd.read_csv(path)

    @abstractmethod
    def get_instructions(self) -> str:
        """Return the task-specific instructions shown to the agent.

        Returns:
            str: A multi-line instruction string.
        """
        ...

    # ── convenience ─────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<{self.__class__.__name__} task_id={self.task_id!r} "
            f"difficulty={self.difficulty!r} max_steps={self.max_steps}>"
        )
