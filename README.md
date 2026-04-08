---
title: OpenEnv Data Cleaning
emoji: 🧹
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# OpenEnv — Data Cleaning Environment

> **LLM agent evaluation environment for real-world CSV data cleaning tasks.**
>
> Built for the [Scaler Meta PyTorch Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon) — OpenEnv Track.

---

## Overview & Motivation

Data cleaning consumes ~80% of a data scientist's time. This environment provides a standardised, reproducible gym where AI agents tackle realistic CSV data-quality issues and receive incremental, programmatic scores.

The environment simulates **real-world data cleaning** — not games or toys:
- Messy CSVs with missing values, type errors, duplicates, outliers, inconsistent categories
- Agents receive observations and submit cleaning actions
- A programmatic grader scores performance incrementally (0.0–1.0)
- Three difficulty levels: easy → medium → hard

**OpenEnv Spec Compliance:**
- `reset()` → initial observation
- `step(action)` → observation, reward, done, info
- `state()` → current internal state
- Typed Pydantic models for Action, Observation, State
- `openenv.yaml` with environment metadata
- FastAPI HTTP server on port 7860

---

## Environment Architecture

```
    Agent (LLM)
    ┌─────────────────────────────────┐
    │  Receives Observation           │
    │  Produces Action (JSON)         │
    │  Uses OpenAI-compatible API     │
    └──────────┬──────────────────────┘
               │ reset() / step()
               ▼
    DataCleaningEnv
    ┌─────────────────────────────────┐
    │  Task Registry                  │
    │  Grader (stateless, per-step)   │
    │  State Tracker                  │
    │  ↕ dirty.csv ↔ clean.csv       │
    └─────────────────────────────────┘
               │
               ▼
    FastAPI Server (port 7860)
    ┌─────────────────────────────────┐
    │  POST /reset                    │
    │  POST /step                     │
    │  GET  /state                    │
    │  GET  /health                   │
    └─────────────────────────────────┘
```

---

## Observation Space

| Field             | Type         | Description                                     |
|-------------------|--------------|-------------------------------------------------|
| `task_id`         | `str`        | Unique identifier for the current task          |
| `instructions`    | `str`        | Human-readable task description                 |
| `data_snapshot`   | `str`        | Current CSV state as string                     |
| `errors_detected` | `list[str]`  | Error descriptions the agent can observe        |
| `current_step`    | `int`        | Steps taken so far                              |
| `max_steps`       | `int`        | Maximum steps allowed                           |
| `done`            | `bool`       | Whether the episode has ended                   |

---

## Action Space

| Field          | Type                  | Description                                |
|----------------|-----------------------|--------------------------------------------|
| `action_type`  | `Literal[...]`        | Cleaning operation (see below)             |
| `row_index`    | `Optional[int]`       | Target row (0-indexed)                     |
| `column_name`  | `Optional[str]`       | Target column name                         |
| `new_value`    | `Optional[Any]`       | Replacement value                          |
| `reasoning`    | `Optional[str]`       | Agent's reasoning                          |

**Action Types:**

| Action              | Description                                       |
|---------------------|---------------------------------------------------|
| `fix_cell`          | Overwrite a cell with a new value                 |
| `fill_missing`      | Fill a NaN/missing cell                           |
| `fix_type`          | Convert value to correct type (e.g. "$1,200" → 1200.0) |
| `drop_row`          | Remove a row                                      |
| `remove_duplicate`  | Remove a duplicate row                            |
| `submit`            | Submit current state as final answer              |

---

## Task Descriptions

### Easy — `easy_missing_values`
- **Dataset:** 10-row employee CSV
- **Errors:** 5 NaN values in `salary`, `department`, `age`
- **Max Steps:** 10
- **Difficulty:** Single error type, small dataset

### Medium — `medium_type_and_duplicates`
- **Dataset:** 30-row sales CSV
- **Errors:** Currency strings, mixed dates, 5 duplicates, negative quantities
- **Max Steps:** 25
- **Difficulty:** Multiple error types

### Hard — `hard_full_pipeline`
- **Dataset:** 100-row transaction CSV
- **Errors:** Missing values, type mismatches, 8 duplicates, outliers, inconsistent categories
- **Max Steps:** 60
- **Difficulty:** All error categories combined

---

## Reward Function Breakdown

Rewards are given **incrementally** per step:

| Event                                      | Reward  |
|--------------------------------------------|---------|
| Correctly filled missing value             | +0.10   |
| Correctly fixed type                       | +0.10   |
| Duplicate correctly removed                | +0.15   |
| Outlier correctly handled                  | +0.20   |
| Wrong overwrite                            | −0.10   |
| Unnecessary action                         | −0.05   |
| Dropping valid row                         | −0.20   |
| Final bonus (≥95% match)                   | +0.20   |

**Final score** = `clamp(cumulative / max_possible, 0.0, 1.0)`

---

## Setup & Installation

```bash
# Clone
git clone <your-repo-url>
cd openenv-data-cleaning

# Install dependencies
pip install -r requirements.txt

# Generate datasets
python scripts/generate_datasets.py

# Validate
python validate.py
```

### Configure Credentials

```bash
cp .env.example .env
# Edit .env — set HF_TOKEN to your HuggingFace API key
```

---

## Running with Docker

```bash
# Build
docker build -t openenv-data-cleaning .

# Run (starts FastAPI server on port 7860)
docker run -p 7860:7860 --env-file .env openenv-data-cleaning

# Verify
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'
```

---

## Running the Baseline Inference Script

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"

# Run inference
python inference.py
```

**Output format** (mandatory for hackathon):
```
[START] task=easy_missing_values env=data-cleaning model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=fill_missing(row=1,col='salary') reward=0.11 done=false error=null
...
[END] success=true steps=6 score=0.82 rewards=0.11,0.11,0.11,0.11,0.11,0.27
```

---

## Baseline Performance Scores

| Task                         | Score | Steps | Status  |
|------------------------------|-------|-------|---------|
| `easy_missing_values`        | —     | —     | —       |
| `medium_type_and_duplicates` | —     | —     | —       |
| `hard_full_pipeline`         | —     | —     | —       |

> Run `python inference.py` with valid credentials to fill these in.

---

## Deploying to HuggingFace Spaces

1. Create a new Space on [HuggingFace](https://huggingface.co/spaces) with **Docker** SDK.
2. Push this repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/openenv-data-cleaning
   git push space main
   ```
3. Set `HF_TOKEN` secret in Space settings.
4. Verify: `curl -X POST https://your-space.hf.space/reset`

---

## Project Structure

```
openenv-data-cleaning/
├── server/
│   ├── app.py               # FastAPI server (POST /reset, /step, /state)
│   └── __init__.py
├── environment/
│   ├── env.py               # Core DataCleaningEnv (reset/step/state)
│   ├── models.py            # Pydantic models (Observation, Action, State, RewardInfo)
│   └── grader.py            # Stateless scoring logic
├── tasks/
│   ├── base_task.py          # Abstract base class
│   ├── task_easy.py          # Easy: fill missing values
│   ├── task_medium.py        # Medium: fix types + duplicates
│   └── task_hard.py          # Hard: full pipeline
├── data/{easy,medium,hard}/  # dirty.csv + clean.csv
├── scripts/
│   └── generate_datasets.py  # Deterministic CSV generator (seed=42)
├── inference.py              # Baseline LLM agent (mandatory format)
├── validate.py               # Local validation
├── openenv.yaml              # OpenEnv spec manifest
├── Dockerfile                # Server container
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## License

MIT
