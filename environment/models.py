"""Pydantic v2 models for the OpenEnv Data Cleaning environment.

Defines the core data contracts: Observation, Action, State, and RewardInfo.
Every interaction between the agent, the environment, and the grader flows
through one of these typed models.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Observation(BaseModel):
    """What the agent sees after each environment step.

    Attributes:
        task_id: Unique identifier for the current task.
        instructions: Human-readable task description and cleaning guidance.
        data_snapshot: Current state of the CSV rendered as a string.
        errors_detected: List of error descriptions the agent can observe.
        current_step: How many steps have been taken so far.
        max_steps: Maximum steps allowed for this task.
        done: Whether the episode has ended.
    """

    task_id: str
    instructions: str
    data_snapshot: str
    errors_detected: list[str] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 10
    done: bool = False


class Action(BaseModel):
    """An action submitted by the agent to the environment.

    Attributes:
        action_type: The kind of cleaning operation being performed.
        row_index: Target row (0-indexed). Required for row-level actions.
        column_name: Target column. Required for cell-level actions.
        new_value: Replacement value for fix_cell / fill_missing / fix_type.
        reasoning: Free-text explanation the agent provides for its choice.
    """

    action_type: Literal[
        "fix_cell",
        "drop_row",
        "fill_missing",
        "fix_type",
        "remove_duplicate",
        "submit",
    ]
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    new_value: Optional[Any] = None
    reasoning: Optional[str] = None

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        """Ensure action_type is one of the allowed literals."""
        allowed = {
            "fix_cell",
            "drop_row",
            "fill_missing",
            "fix_type",
            "remove_duplicate",
            "submit",
        }
        if v not in allowed:
            raise ValueError(f"action_type must be one of {allowed}, got '{v}'")
        return v


class State(BaseModel):
    """Internal environment state exposed via env.state().

    Attributes:
        task_id: Current task identifier.
        steps_taken: Number of steps consumed so far.
        total_errors: How many errors existed in the original dirty data.
        errors_fixed: How many of those errors the agent has correctly fixed.
        errors_introduced: Correct values that the agent overwrote (penalised).
        score_so_far: Cumulative normalised score in [0.0, 1.0].
        is_complete: Whether the episode is finished.
    """

    task_id: str
    steps_taken: int = 0
    total_errors: int = 0
    errors_fixed: int = 0
    errors_introduced: int = 0
    score_so_far: float = 0.0
    is_complete: bool = False


class RewardInfo(BaseModel):
    """Detailed reward breakdown returned by the grader.

    Attributes:
        step_reward: Reward earned in this single step.
        cumulative_reward: Total reward accumulated across all steps.
        reward_breakdown: Component-wise breakdown (e.g. fix_accuracy, penalty).
        feedback: Human-readable explanation of the reward.
    """

    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    feedback: str = ""
