# 📬 Email Triage — OpenEnv AI Environment

A **real-world AI training environment** where an agent learns to triage a corporate inbox through progressively harder tasks: classification, prioritisation, routing, and drafting compliant customer replies.

---

## 🧠 What is this project?

This is **not a traditional app** — it is an **OpenEnv-compliant simulation environment**.

It allows an AI agent to:
- Observe an inbox
- Take structured actions
- Receive **step-wise rewards**
- Improve performance over time

👉 Think of it as a **gym for AI agents** to learn real-world email handling.

---

## 🎯 Why email triage?

Email triage is a canonical real-world task:

- Professionals spend ~28% of their time on email
- Requires **multi-step reasoning**
- Involves **policy compliance + tone control**
- Has **clear ground truth → objective evaluation**
- Scales from simple classification → complex generation

---

## ⚙️ Core idea

The environment simulates:

```
Inbox → Agent Action → Evaluation → Reward → Next Step
```

This enables:
- Reinforcement learning-style training
- Deterministic benchmarking of LLMs
- Structured evaluation of agent behavior

---

## 📂 Project structure

```
email-triage-env/
├── openenv.yaml          # OpenEnv specification
├── Dockerfile            # Container setup (port 7860)
├── requirements.txt      # Dependencies
├── client.py             # Python API client
├── inference.py          # Baseline LLM agent
│
├── env/
│   ├── environment.py    # Core OpenEnv interface (reset/step/state)
│   ├── tasks.py          # Task logic + reward functions
│   ├── data.py           # Synthetic dataset + ground truth
│   └── models.py         # Pydantic schemas
│
└── api/
    └── main.py           # FastAPI server
```

---

## 🧩 Tasks

| Task | Description | Difficulty |
|------|------------|-----------|
| `task1_classify` | Classify emails into categories | Easy |
| `task2_prioritise` | Assign urgency + route to correct queue | Medium |
| `task3_draft_reply` | Generate policy-compliant replies | Hard |

---

## 🏆 Reward design

### Task 1 — Classification
- ✅ Exact match → **1.0**
- ⚖️ Adjacent category → **0.5**
- ❌ Wrong → **0.0**

---

### Task 2 — Prioritise & Route
Per email:
- Urgency → **0.4**
- Routing → **0.6**

Partial credit for near-correct urgency.

---

### Task 3 — Draft Reply
Multi-component reward:
- Keywords → **0.4**
- Policy compliance → **0.4**
- Forbidden phrases → **−0.1 each**
- No actionable commitment → **−0.1**

All rewards are **dense (step-wise)**.

---

## 🤖 Baseline agent

Runs via `inference.py` using an OpenAI-compatible client.

**Default model:**
```
Qwen/Qwen2.5-72B-Instruct
```

### Sample performance

| Task | Score |
|------|------|
| Classification | ~0.9 |
| Prioritise | ~0.8–0.9 |
| Draft Reply | ~0.6–0.7 |

---

## 🔌 API

| Method | Endpoint | Description |
|--------|--------|-------------|
| GET | `/health` | Health check |
| GET | `/tasks` | Available tasks |
| GET | `/state` | Current state |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take action |

---

## 🚀 Setup

### 1. Clone

```bash
git clone <repo-url>
cd email-triage-env
```

---

### 2. Install

```bash
pip install -r requirements.txt
```

---

### 3. Set environment variables

Create `.env`:

```env
HF_TOKEN=your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

---

### 4. Run server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

---

### 5. Run agent

```bash
python inference.py
```

---

## 📊 Output format (strict)

```
[START] task=... env=email-triage model=...
[STEP] step=... action=... reward=... done=...
[END] success=... steps=... rewards=...
```

---

## 🧪 Example usage

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "task1_classify"}'
```

---

## 🧠 Design choices

### Synthetic dataset
- 10 curated emails
- Covers support, billing, spam, engineering, sales
- Includes realistic edge cases

### Internal email IDs
- IDs like `e001`, `e002` are **internal identifiers**
- Not real email addresses (by design)

### No name assumptions
- Agent instructed to avoid hallucinating names
- Uses neutral addressing (e.g. "Customer")

---

## 🔧 Extensibility

- Add emails → `env/data.py`
- Add tasks → `env/tasks.py`
- Swap models → change env variables
- Plug into RL pipelines → OpenEnv compatible

---

## 💡 Why this matters

This project demonstrates:

- Structured evaluation of LLM agents
- Real-world task simulation
- Multi-step reasoning with feedback
- Policy-aware text generation

---

## 🏁 Summary

> A scalable, realistic environment where AI agents learn to handle email workflows with measurable performance.

---

## 📜 License

MIT
