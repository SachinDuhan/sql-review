import uuid
from typing import Optional
from .models import SQLAction, SQLObservation, SQLState
from .tasks import TASKS, GRADERS

MAX_STEPS = 5


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class SQLReviewEnvironment:
    def __init__(self, task_name: str = "fix_syntax_error"):
        self.task_name = task_name
        self._episode_id: str = ""
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = False
        self._last_submission: Optional[str] = None
        self._last_feedback: Optional[str] = None
        self._best_score: float = 0.0

    def reset(self) -> SQLObservation:
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._last_submission = None
        self._last_feedback = None
        self._best_score = 0.0

        task = TASKS[self.task_name]
        return SQLObservation(
            task_name=self.task_name,
            broken_query=task["broken_query"].strip(),
            schema_context=task["schema_context"].strip(),
            error_description=task["error_description"],
            hint=None,
            done=False,
            reward=0.0,
            step_count=0,
        )

    def step(self, action: SQLAction):
        if self._done:
            task = TASKS[self.task_name]
            obs = SQLObservation(
                task_name=self.task_name,
                broken_query=task["broken_query"].strip(),
                schema_context=task["schema_context"].strip(),
                error_description=task["error_description"],
                last_submission=self._last_submission,
                last_feedback="Episode already done.",
                done=True,
                reward=0.0,
                step_count=self._step_count,
            )
            return obs, 0.0, True, {}

        self._step_count += 1
        grader = GRADERS[self.task_name]
        score, feedback = grader(action.fixed_query)
        score = _clamp01(score)

        # Reward shaping in [0, 1]: improvement-based with bounded no-progress penalty.
        improvement = max(0.0, score - self._best_score)
        if improvement > 0:
            step_reward = min(1.0, 0.2 + 0.8 * improvement)
        else:
            # Keep reward in range while penalizing repeated non-progress.
            step_reward = max(0.0, score - 0.1)

        step_reward = _clamp01(step_reward)

        self._best_score = max(self._best_score, score)
        self._total_reward += step_reward
        self._last_submission = action.fixed_query
        self._last_feedback = feedback

        # Done if perfect score or max steps reached
        if score >= 1.0 or self._step_count >= MAX_STEPS:
            self._done = True

        task = TASKS[self.task_name]
        obs = SQLObservation(
            task_name=self.task_name,
            broken_query=task["broken_query"].strip(),
            schema_context=task["schema_context"].strip(),
            error_description=task["error_description"],
            hint=task.get("hint") if self._step_count >= 2 else None,
            last_submission=action.fixed_query,
            last_feedback=feedback,
            done=self._done,
            reward=step_reward,
            step_count=self._step_count,
        )
        return obs, step_reward, self._done, {"score": score, "feedback": feedback}

    def state(self) -> SQLState:
        return SQLState(
            episode_id=self._episode_id,
            task_name=self.task_name,
            step_count=self._step_count,
            total_reward=self._total_reward,
            done=self._done,
            max_steps=MAX_STEPS,
        )