"""FastAPI application for the OpenEnv Data Cleaning environment.

Exposes the environment over HTTP endpoints compatible with
the OpenEnv specification and HuggingFace Spaces.

Endpoints:
    POST /reset         Reset the environment (accepts optional task_id)
    POST /step          Execute an action
    GET  /state         Get current environment state
    GET  /health        Health check
    GET  /              Root — environment info

Usage::

    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment.env import DataCleaningEnv
from environment.models import Action, Observation, State

# ── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(
    title="OpenEnv Data Cleaning",
    description="LLM agent evaluation environment for real-world CSV data cleaning tasks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global environment instance ─────────────────────────────────────

env = DataCleaningEnv()


# ── Request / Response schemas ──────────────────────────────────────

class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    task_id: str = "easy_missing_values"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    """Request body for POST /step."""
    action: dict


class ResetResponse(BaseModel):
    """Response body for POST /reset."""
    observation: dict
    reward: float = 0.0
    done: bool = False
    metadata: dict = {}


class StepResponse(BaseModel):
    """Response body for POST /step."""
    observation: dict
    reward: float
    done: bool
    metadata: dict = {}


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    environment: str
    version: str


# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict:
    """Root endpoint — environment info."""
    return {
        "name": "openenv-data-cleaning",
        "version": "1.0.0",
        "description": "LLM agent evaluation environment for real-world CSV data cleaning tasks",
        "tasks": ["easy_missing_values", "medium_type_and_duplicates", "hard_full_pipeline"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint — returns 200 if the server is alive."""
    return HealthResponse(
        status="healthy",
        environment="data-cleaning",
        version="1.0.0",
    )


@app.post("/reset")
async def reset(request: ResetRequest = None) -> ResetResponse:
    """Reset the environment and return the initial observation.

    The validator pings this endpoint with POST {} and expects HTTP 200.
    """
    if request is None:
        request = ResetRequest()

    try:
        obs: Observation = env.reset(request.task_id)
        return ResetResponse(
            observation=obs.model_dump(),
            reward=0.0,
            done=obs.done,
            metadata={
                "task_id": obs.task_id,
                "max_steps": obs.max_steps,
                "errors_detected": len(obs.errors_detected),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
async def step(request: StepRequest) -> StepResponse:
    """Execute a single action in the environment."""
    try:
        action = Action(**request.action)
        obs, reward, done, info = env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=reward,
            done=done,
            metadata=info,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
async def state() -> dict:
    """Return the current internal state of the environment."""
    try:
        s: State = env.state()
        return s.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Entrypoint ──────────────────────────────────────────────────────

def main():
    """Run the server directly (for development or Docker CMD)."""
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
