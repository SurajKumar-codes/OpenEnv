"""
OpenEnv Data Cleaning — Baseline Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from environment.env import DataCleaningEnv
from environment.models import Action

# ── Configuration (MANDATORY env vars) ──────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "data-cleaning"
TASKS = [
    "easy_missing_values",
    "medium_type_and_duplicates",
    "hard_full_pipeline",
]

TEMPERATURE = 0.1
MAX_TOKENS = 512


# ── Structured stdout logging ──────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit [STEP] line."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    reward = max(0.01, min(0.99, reward))
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """Emit [END] line."""
    # Safety clamp: score must be strictly between 0 and 1
    score = max(0.01, min(0.99, score))
    rewards = [max(0.01, min(0.99, r)) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM interaction ────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a data-cleaning AI agent. You interact with a CSV dataset by
    submitting one cleaning action per turn as a JSON object.

    Valid action_type values:
      fix_cell, drop_row, fill_missing, fix_type, remove_duplicate, submit

    Response format — respond with ONLY a JSON object like:
    {"action_type": "fill_missing", "row_index": 1, "column_name": "salary", "new_value": 55000.0, "reasoning": "Filling with median salary"}

    When you believe the data is clean, use: {"action_type": "submit"}
""")


def build_user_prompt(obs_dict: dict) -> str:
    """Build the user prompt from the current observation."""
    errors_block = "\n".join(f"  - {e}" for e in obs_dict.get("errors_detected", []))
    csv_snapshot = obs_dict.get("data_snapshot", "")[:3000]

    return textwrap.dedent(f"""\
        ## Task Instructions
        {obs_dict.get('instructions', 'Clean the dataset.')}

        ## Current CSV (first 3000 chars)
        ```
        {csv_snapshot}
        ```

        ## Detected Errors
        {errors_block if errors_block else "  None detected."}

        Step {obs_dict.get('current_step', 0)} / {obs_dict.get('max_steps', 10)}

        Respond with a single JSON action object.
    """)


def get_model_action(client: OpenAI, obs_dict: dict) -> Action:
    """Call the LLM and parse the response as an Action."""
    user_prompt = build_user_prompt(obs_dict)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return Action(action_type="submit", reasoning="LLM call failed")

    return parse_action(text)


def parse_action(text: str) -> Action:
    """Extract a JSON Action from the LLM response text."""
    json_match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return Action(**data)
        except Exception:
            pass

    return Action(action_type="submit", reasoning="Could not parse LLM output")


# ── Task runner ─────────────────────────────────────────────────────

def run_task(client: OpenAI, env: DataCleaningEnv, task_id: str) -> dict:
    """Run a single task and return results dict.

    Emits [START], [STEP]..., [END] to stdout.
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id)
        done = obs.done
        max_steps = obs.max_steps

        for step_num in range(1, max_steps + 1):
            if done:
                break

            action = get_model_action(client, obs.model_dump())

            obs, reward, done, info = env.step(action)

            # Build action string for logging
            action_str = f"{action.action_type}"
            if action.row_index is not None:
                action_str += f"(row={action.row_index}"
                if action.column_name:
                    action_str += f",col='{action.column_name}'"
                action_str += ")"

            error = None  # Our env doesn't produce last_action_error

            rewards.append(reward)
            steps_taken = step_num

            log_step(
                step=step_num,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # Compute final score (use the last cumulative reward, clamped strictly between 0 and 1)
        final_state = env.state()
        score = max(0.01, min(0.99, final_state.score_so_far))
        success = score >= 0.1  # threshold for "success"

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)
        score = 0.01
        success = False

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return {
        "task_id": task_id,
        "score": round(score, 2),
        "steps": steps_taken,
        "success": success,
        "rewards": rewards,
    }


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    """Run inference on all tasks sequentially."""
    if not HF_TOKEN:
        print(
            "[ERROR] HF_TOKEN is not set. "
            "Set it in your .env file or export it.",
            flush=True,
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = DataCleaningEnv()

    all_results: List[dict] = []

    for task_id in TASKS:
        result = run_task(client, env, task_id)
        all_results.append(result)

    # Save results.json
    results_path = Path("results.json")
    results_path.write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
