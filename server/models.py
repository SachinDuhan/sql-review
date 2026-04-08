from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class SQLAction(BaseModel):
    """The action an agent takes: submit a fixed SQL query."""
    fixed_query: str
    explanation: Optional[str] = None

class SQLObservation(BaseModel):
    """What the agent sees after each step."""
    task_name: str
    broken_query: str
    schema_context: str
    error_description: str
    hint: Optional[str] = None
    last_submission: Optional[str] = None
    last_feedback: Optional[str] = None
    done: bool = False
    reward: float = 0.0
    step_count: int = 0


class SQLReward(BaseModel):
    """Step reward normalized to [0.0, 1.0]."""
    value: float = Field(ge=0.0, le=1.0)


class SQLStepResponse(BaseModel):
    """Typed response for step(action)."""
    observation: SQLObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any]

class SQLState(BaseModel):
    """Internal episode state."""
    episode_id: str
    task_name: str
    step_count: int
    total_reward: float
    done: bool
    max_steps: int