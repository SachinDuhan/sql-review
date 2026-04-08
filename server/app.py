import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import SQLAction, SQLObservation, SQLState, SQLStepResponse
from .environment import SQLReviewEnvironment
from .tasks import TASKS
from typing import Optional

app = FastAPI(
    title="SQL Review OpenEnv",
    description="OpenEnv environment for SQL query review and repair.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TASK_NAME = os.getenv("SQL_REVIEW_TASK", "fix_syntax_error")
_env = SQLReviewEnvironment(task_name=TASK_NAME)


@app.post("/reset", response_model=SQLObservation)
def reset(task: Optional[str] = None):
    global _env, TASK_NAME
    t = task or TASK_NAME
    if t not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {t}")
    _env = SQLReviewEnvironment(task_name=t)
    return _env.reset()


@app.post("/step", response_model=SQLStepResponse)
def step(action: SQLAction):
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=SQLState)
def state():
    return _env.state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": name,
                "difficulty": data["difficulty"],
                "error_description": data["error_description"],
            }
            for name, data in TASKS.items()
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()