"""
inference.py — SQL Review OpenEnv Baseline Inference Script
Runs all 3 tasks and emits [START]/[STEP]/[END] logs.
"""

import os
import sys
import textwrap
from typing import List, Optional
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_KEY = OPENAI_API_KEY or HF_TOKEN
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS = 5
SUCCESS_THRESHOLD = 0.6

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = ["fix_syntax_error", "fix_logic_error", "optimize_query"]

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert SQL engineer. You will be given a broken SQL query and must fix it.
Read the error description carefully and submit a corrected SQL query.

Rules:
- Output ONLY the fixed SQL query, no explanation, no markdown, no backticks.
- Do not add comments.
- Keep the same general intent of the original query.
- Fix all issues described.
""").strip()


def safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except BrokenPipeError:
        # Exit quietly if output is being piped and downstream closes early (e.g. head).
        sys.exit(0)


def log_start(task, env, model):
    safe_print(f"[START] task={task} env={env} model={model}")

def log_step(step, action, reward, done, error):
    action_clean = action.replace('\n', ' ').strip()[:120]
    error_val = error if error else "null"
    safe_print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={str(done).lower()} error={error_val}")

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    safe_print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}")


def call_llm(broken_query: str, schema: str, error_desc: str,
             last_submission: Optional[str], last_feedback: Optional[str],
             hint: Optional[str]) -> str:
    parts = [
        f"Schema:\n{schema}",
        f"Broken Query:\n{broken_query}",
        f"Problem:\n{error_desc}",
    ]
    if hint:
        parts.append(f"Hint: {hint}")
    if last_submission:
        parts.append(f"Your last attempt:\n{last_submission}")
    if last_feedback:
        parts.append(f"Feedback on last attempt: {last_feedback}")
    parts.append("Write the corrected SQL query:")

    user_prompt = "\n\n".join(parts)

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.split("\n")
                if not line.startswith("```")
            ).strip()
        return text or "SELECT 1;"
    except Exception:
        return "SELECT 1;"


def run_task(task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    best_score = 0.0

    log_start(task=task_name, env="sql-review-env", model=MODEL_NAME)

    # Reset env with correct task
    try:
        resp = requests.post(f"{ENV_URL}/reset",
                             params={"task": task_name}, timeout=15)
        resp.raise_for_status()
        obs = resp.json()
    except Exception:
        log_end(False, 0, 0.0, [])
        return 0.0

    for step in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            break

        fixed_sql = call_llm(
            broken_query=obs.get("broken_query", ""),
            schema=obs.get("schema_context", ""),
            error_desc=obs.get("error_description", ""),
            last_submission=obs.get("last_submission"),
            last_feedback=obs.get("last_feedback"),
            hint=obs.get("hint"),
        )

        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"fixed_query": fixed_sql, "explanation": ""},
                timeout=15,
            )
            step_resp.raise_for_status()
            result = step_resp.json()
        except Exception as e:
            log_step(step, fixed_sql, 0.0, False, str(e))
            break

        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        info = result.get("info", {})
        obs = result.get("observation", obs)
        error = None
        best_score = max(best_score, float(info.get("score", 0.0)))

        rewards.append(reward)
        steps_taken = step
        log_step(step, fixed_sql, reward, done, error)

        if done:
            score = float(info.get("score", best_score))
            success = score >= SUCCESS_THRESHOLD
            break

    if not rewards:
        score = 0.0
    elif score == 0.0:
        # fall back to best grader score seen during the episode
        score = best_score

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def main():
    all_scores = []
    for task in TASKS:
        # Restart env with this task by setting env var isn't possible at runtime,
        # so we patch via query param (handled in reset endpoint enhancement below)
        score = run_task(task)
        all_scores.append(score)
        safe_print(f"# Task {task}: score={score:.3f}")

    avg = sum(all_scores) / len(all_scores)
    safe_print(f"# Average score across all tasks: {avg:.3f}")


if __name__ == "__main__":
    main()