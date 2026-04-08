# 📬 Email Triage — OpenEnv Environment

An OpenEnv-compliant environment where an AI agent learns to triage a
corporate inbox through three progressively harder tasks: classifying emails,
prioritising and routing them to the right queues, and drafting
policy-compliant customer replies.

---

## Why email triage?

Email triage is a canonical real-world task:
- Humans spend ~28 % of the working week reading and answering email
- It requires multi-step reasoning, policy compliance, and tone awareness
- It has clear ground truth enabling objective, deterministic grading
- It scales naturally from simple classification to open-ended generation

---

## Project structure

```
email-triage-env/
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile            # Container build (port 7860)
├── requirements.txt      # Runtime dependencies
├── client.py             # HTTP client (EnvClient)
├── inference.py          # Baseline LLM agent
│
├── env/
│   ├── __init__.py
│   ├── environment.py    # EmailTriageEnv  (reset / step / state / close)
│   ├── tasks.py          # Task1Classify, Task2Prioritise, Task3DraftReply + graders
│   ├── data.py           # 10 synthetic emails, ground-truth labels, reference replies
│   └── models.py         # Pydantic models: Email, EmailTriageAction, EmailTriageObservation
│
└── api/
    └── main.py           # FastAPI app  (/reset /step /state /tasks /health)
```

---

## Tasks

| ID | Name | Difficulty | Agent action | Max steps |
|----|------|-----------|-------------|-----------|
| `task1_classify` | Email Classification | Easy | Classify each email: `spam` / `billing` / `support` / `sales` / `engineering` | 6 |
| `task2_prioritise` | Prioritise & Route | Medium | Alternate: assign urgency (`low`/`medium`/`high`/`critical`) then route to correct queue | 12 |
| `task3_draft_reply` | Draft Reply | Hard | Write a policy-compliant reply (apologise, give ETA, no blame, actionable) | 2 |

---

## Reward functions

| Task | Logic |
|------|-------|
| `task1_classify` | **1.0** exact match · **0.5** semantically adjacent (e.g. `engineering` ↔ `support`) · **0.0** otherwise |
| `task2_prioritise` | Urgency correct **+0.4** · off by one level **+0.2** · route correct **+0.6** · per email max **1.0** |
| `task3_draft_reply` | Keywords present **+0.4** · policy rules (apologise, ETA, no-blame) **+0.4** · forbidden phrases **−0.1 each** · no actionable commitment **−0.1** · clamped to **[0.0, 1.0]** |

All rewards are **dense** (provided every step, not just at episode end).  
`skip` actions carry an additional **−0.2** penalty.

---

## Baseline scores (Qwen/Qwen2.5-72B-Instruct via HF router)

| Task | Score |
|------|-------|
| `task1_classify` | ~0.83 |
| `task2_prioritise` | ~0.67 |
| `task3_draft_reply` | ~0.60 |

---

## Action space

```json
{
  "action_type": "classify | prioritize | route | draft_reply | skip",
  "email_id":   "<string>",
  "value":      "<string>"
}
```

| `action_type` | valid `value` |
|---------------|--------------|
| `classify`    | `spam` `billing` `support` `sales` `engineering` |
| `prioritize`  | `low` `medium` `high` `critical` |
| `route`       | `spam` `billing` `support` `sales` `engineering` |
| `draft_reply` | Free-text reply string |
| `skip`        | Reason string (−0.2 penalty) |

---

## Observation space

```json
{
  "emails":           [...],
  "current_email":    {"id": "e001", "subject": "...", "body": "...", "sender": "...", "timestamp": "..."},
  "step":             0,
  "task_description": "...",
  "context": {
    "valid_categories": [...],
    "expected_action":  "prioritize | route",
    "feedback":         "...",
    "policy":           "..."
  }
}
```

---

## Setup and usage

### Prerequisites

- Python ≥ 3.11
- A HuggingFace token (or any OpenAI-compatible API key)

### Local — Python

```bash
git clone <repo-url>
cd email-triage-env

pip install -r requirements.txt

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 7860 --reload
```

The server is ready when `/health` returns `{"status": "ok"}`.

```bash
# In a second terminal — run the baseline agent
export HF_TOKEN=hf_...
python inference.py
```

### Docker

```bash
docker build -t email-triage-env .

docker run -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  email-triage-env
```

---

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Returns `{"status": "ok", "env": "email-triage", "version": "1.0.0"}` |
| `GET`  | `/tasks`  | Lists available task names |
| `GET`  | `/state`  | Current env state (steps, score, reward history) |
| `POST` | `/reset`  | Start a new episode |
| `POST` | `/step`   | Submit one action |
| `GET`  | `/`       | Index / discovery |

### Quick-start with curl

```bash
# Reset to task 1
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "task1_classify"}' | jq .

# Classify an email
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "email_id": "e001", "value": "spam"}' | jq .

# Check state
curl -s http://localhost:7860/state | jq .
```

---

## Using the Python client

```python
from client import EnvClient

client = EnvClient()                  # default: http://localhost:7860
client.wait_until_healthy()

# Reset
resp = client.reset("task1_classify")
obs  = resp["observation"]

# Step loop
while not obs.get("done"):
    email_id = obs["current_email"]["id"]
    result   = client.step("classify", email_id, "support")
    obs      = result["observation"]
    print(result["reward"], result["done"], result["info"]["feedback"])

print("Score:", client.state()["score"])
```

---

## Running inference.py

```bash
export HF_TOKEN=hf_...
# Optional overrides:
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

### Log output format

```
[START] task=task1_classify env=email-triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type": "classify", "email_id": "e001", "value": "spam"} reward=1.00 done=false error=null
[STEP] step=2 action={"action_type": "classify", "email_id": "e003", "value": "billing"} reward=1.00 done=false error=null
...
[END] success=true steps=6 score=0.833 rewards=1.00,1.00,1.00,0.50,1.00,1.00

[START] task=task2_prioritise env=email-triage model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=12 score=0.667 rewards=...

[START] task=task3_draft_reply env=email-triage model=Qwen/Qwen2.5-72B-Instruct
...
[END] success=true steps=2 score=0.600 rewards=...
```

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace / API key (also accepted as `API_KEY`) |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` | No | `http://localhost:7860` | Server URL used by `client.py` |

---

## Extending the environment

**Add emails** — edit `env/data.py`: append to `EMAILS` and add entries to `GROUND_TRUTH` (and `REFERENCE_REPLIES` for task 3).

**Add a task** — create a new `TaskN` class in `env/tasks.py` following the same `reset() / step() / score()` pattern, then register it in `TASKS`.

**Swap the LLM** — set `API_BASE_URL` and `MODEL_NAME` to any OpenAI-compatible endpoint (Together AI, Fireworks, local vLLM, etc.).

---

## License

MIT
