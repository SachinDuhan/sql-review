---
title: SQL Review OpenEnv
emoji: "🐳"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# SQL Review OpenEnv

An OpenEnv environment where an AI agent reviews and repairs broken SQL queries.
Simulates real-world data engineering tasks.

## Environment Description

The agent receives a broken SQL query, a schema, and an error description. It must
submit fixed SQL. Rewards are shaped to signal partial progress while remaining in [0.0, 1.0].

## Action Space
```json
{ "fixed_query": "<SQL string>", "explanation": "<optional string>" }
```

## Observation Space
- `broken_query`: The original faulty SQL
- `schema_context`: Relevant table/column definitions
- `error_description`: Description of the bug
- `hint`: Unlocked after step 2
- `last_submission`: Agent's previous attempt
- `last_feedback`: Grader feedback on previous attempt
- `done`, `reward`, `step_count`

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `fix_syntax_error` | Easy | HAVING clause uses alias — fix to reference aggregate |
| `fix_logic_error` | Medium | WHERE on LEFT JOIN silently becomes INNER JOIN |
| `optimize_query` | Hard | Replace slow IN-subquery with JOIN; fix COUNT logic |

## Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t sql-review-env .
docker run -p 7860:7860 sql-review-env
```

## Inference

```bash
export OPENAI_API_KEY=your_token  # preferred
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_URL=http://localhost:7860
python inference.py
```

The baseline script supports both OPENAI_API_KEY and HF_TOKEN, and uses deterministic
sampling (temperature=0.0) for reproducible evaluation runs.

## Baseline Scores (Qwen2.5-72B)
- fix_syntax_error: ~0.85
- fix_logic_error: ~0.70
- optimize_query: ~0.55
# sql-review
