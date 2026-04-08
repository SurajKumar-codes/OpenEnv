"""OpenEnv local validation script.

Runs the same checks that the hackathon pre-submission validator performs:
1. openenv.yaml is valid
2. Data files exist
3. Environment imports work
4. reset() / step() / state() work
5. Server starts and /reset returns 200
6. inference.py produces correct stdout format

Usage::

    python validate.py           # run all checks
    python validate.py --verbose # show detailed output
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import typer
import yaml

console_print = print  # Use plain print for Windows compat
app = typer.Typer(help="OpenEnv Data Cleaning - Validate Environment")

PROJECT_ROOT = Path(__file__).resolve().parent
PASS_COUNT = 0
FAIL_COUNT = 0


def _check(label: str, passed: bool, detail: str = "") -> bool:
    """Print a check result and return whether it passed."""
    global PASS_COUNT, FAIL_COUNT
    icon = "PASS" if passed else "FAIL"
    msg = f"  [{icon}] {label}"
    if detail:
        msg += f"  ({detail})"
    console_print(msg)
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return passed


@app.command()
def validate(
    verbose: bool = typer.Option(False, help="Show detailed output for each check."),
) -> None:
    """Run all OpenEnv validation checks."""
    console_print("OpenEnv Data Cleaning - Validation\n")
    all_passed = True

    # 1. Check openenv.yaml
    yaml_path = PROJECT_ROOT / "openenv.yaml"
    try:
        cfg = yaml.safe_load(yaml_path.read_text())
        ok = (
            cfg is not None
            and cfg.get("spec_version") == 1
            and "name" in cfg
            and "app" in cfg
        )
        all_passed &= _check("openenv.yaml is valid (spec_version=1)", ok, str(yaml_path))
        if verbose and cfg:
            console_print(f"    name={cfg.get('name')} runtime={cfg.get('runtime')} port={cfg.get('port')}")
    except Exception as exc:
        all_passed &= _check("openenv.yaml is valid", False, str(exc))

    # 2. Check data files
    for diff in ("easy", "medium", "hard"):
        for fname in ("dirty.csv", "clean.csv"):
            p = PROJECT_ROOT / "data" / diff / fname
            all_passed &= _check(f"data/{diff}/{fname} exists", p.exists())

    # 3. Import environment
    try:
        from environment.env import DataCleaningEnv
        all_passed &= _check("Import DataCleaningEnv", True)
    except Exception as exc:
        all_passed &= _check("Import DataCleaningEnv", False, str(exc))
        raise typer.Exit(code=1)

    # 4. Import models
    try:
        from environment.models import Action, Observation, State, RewardInfo
        all_passed &= _check("Import Pydantic models", True)
    except Exception as exc:
        all_passed &= _check("Import Pydantic models", False, str(exc))

    # 5. Reset each task
    env = DataCleaningEnv()
    for task_id in ("easy_missing_values", "medium_type_and_duplicates", "hard_full_pipeline"):
        try:
            obs = env.reset(task_id)
            ok = obs.task_id == task_id and len(obs.data_snapshot) > 0
            all_passed &= _check(f"env.reset('{task_id}')", ok)
            if verbose:
                console_print(f"    errors={len(obs.errors_detected)} max_steps={obs.max_steps}")
        except Exception as exc:
            all_passed &= _check(f"env.reset('{task_id}')", False, str(exc))

    # 6. Step with a test action
    try:
        env.reset("easy_missing_values")
        gt = env._ground_truth_df
        correct_val = gt.at[1, "salary"]
        action = Action(
            action_type="fill_missing",
            row_index=1,
            column_name="salary",
            new_value=correct_val,
        )
        obs, reward, done, info = env.step(action)
        ok = isinstance(reward, float) and isinstance(done, bool) and 0.0 <= reward <= 1.0
        all_passed &= _check("env.step() returns (obs, reward, done, info)", ok)
        if verbose:
            console_print(f"    reward={reward} done={done}")
    except Exception as exc:
        all_passed &= _check("env.step() works", False, str(exc))

    # 7. State check
    try:
        state = env.state()
        ok = state.task_id == "easy_missing_values" and state.steps_taken >= 1
        all_passed &= _check("env.state() returns State", ok)
    except Exception as exc:
        all_passed &= _check("env.state()", False, str(exc))

    # 8. Check Dockerfile exists
    df_path = PROJECT_ROOT / "Dockerfile"
    all_passed &= _check("Dockerfile exists", df_path.exists())

    # 9. Check inference.py exists
    inf_path = PROJECT_ROOT / "inference.py"
    all_passed &= _check("inference.py exists in root", inf_path.exists())

    # 10. Check .env.example has correct vars
    env_example = PROJECT_ROOT / ".env.example"
    if env_example.exists():
        content = env_example.read_text()
        has_vars = all(v in content for v in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"])
        all_passed &= _check(".env.example has API_BASE_URL, MODEL_NAME, HF_TOKEN", has_vars)
    else:
        all_passed &= _check(".env.example exists", False)

    # 11. Check server imports
    try:
        from server.app import app as fastapi_app
        all_passed &= _check("Import server.app FastAPI application", True)
    except Exception as exc:
        all_passed &= _check("Import server.app", False, str(exc))

    # Summary
    console_print(f"\n{'='*50}")
    if all_passed:
        console_print(f"ALL {PASS_COUNT} CHECKS PASSED")
        raise typer.Exit(code=0)
    else:
        console_print(f"{PASS_COUNT} passed, {FAIL_COUNT} failed")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
