"""
Inference Script — Email Triage OpenEnv
=======================================
Runs a baseline LLM agent against all 3 tasks and emits structured logs.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

load_dotenv()

HF_TOKEN         = os.getenv("HF_TOKEN")
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "email-triage"

MAX_STEPS_PER_TASK = 20
SUCCESS_THRESHOLD  = 0.5   
TEMPERATURE        = 0.2
MAX_TOKENS         = 150

# ---------------------------------------------------------------------------
# Logging helpers (STRICT FORMAT)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_safe = action.replace("\n", " ").replace("\r", "")[:200]

    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: Dict[str, str] = {
    "task1_classify": textwrap.dedent("""
        You are an email triage assistant.

        Classify the email into ONE of:
        spam, billing, support, sales, engineering.

        Respond ONLY with JSON:
        {"action_type": "classify", "email_id": "<id>", "value": "<category>"}
    """).strip(),

    "task2_prioritise": textwrap.dedent("""
        You are an email triage assistant.

        You must follow the expected_action strictly.

        If expected_action = "prioritize":
        {"action_type": "prioritize", "email_id": "<id>", "value": "<urgency>"}

        If expected_action = "route":
        {"action_type": "route", "email_id": "<id>", "value": "<queue>"}

        Valid urgencies: low, medium, high, critical
        Valid queues: spam, billing, support, sales, engineering

        Respond ONLY with JSON.
    """).strip(),

    "task3_draft_reply": textwrap.dedent("""
        You are a professional support agent.

        MUST include:
        - apology (sorry / apologize)
        - ETA (hours/days)
        - action (we will / follow up / resolve)

        NEVER blame the user.

        Respond ONLY with JSON:
        {"action_type": "draft_reply", "email_id": "<id>", "value": "<reply>"}
    """).strip(),
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Dict[str, Any]) -> str:
    current  = obs.get("current_email")
    context  = obs.get("context", {})
    task_desc = obs.get("task_description", "")

    lines = [f"Task: {task_desc}", ""]

    if current:
        lines += [
            "Email:",
            f"ID: {current['id']}",
            f"Subject: {current['subject']}",
            f"From: {current['sender']}",
            f"Body: {current['body']}",
        ]

    if context.get("expected_action"):
        lines.append(f"Expected action: {context['expected_action']}")

    if context.get("feedback"):
        lines.append(f"Feedback: {context['feedback']}")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_action(client: OpenAI, task_name: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = SYSTEM_PROMPTS[task_name]
    user_prompt   = build_user_prompt(obs)

    text = ""
    action = None

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        text = (completion.choices[0].message.content or "").strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            action = json.loads(match.group())

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)

    # -------------------------------
    # FALLBACK if parsing failed
    # -------------------------------
    if not action:
        current = obs.get("current_email", {})
        email_id = current.get("id", "unknown")

        if task_name == "task1_classify":
            return {
                "action_type": "classify",
                "email_id": email_id,
                "value": "support",
            }

        elif task_name == "task2_prioritise":
            expected = obs.get("context", {}).get("expected_action", "prioritize")

            if expected == "prioritize":
                return {
                    "action_type": "prioritize",
                    "email_id": email_id,
                    "value": "medium",
                }
            else:
                return {
                    "action_type": "route",
                    "email_id": email_id,
                    "value": "support",
                }

        elif task_name == "task3_draft_reply":
            return {
                "action_type": "draft_reply",
                "email_id": email_id,
                "value": "Sorry for the inconvenience. We are looking into this and will follow up within 24 hours.",
            }

    # -------------------------------
    # ENFORCE VALID ACTIONS
    # -------------------------------
    if task_name == "task2_prioritise":
        expected = obs.get("context", {}).get("expected_action")

        if expected == "prioritize":
            action["action_type"] = "prioritize"
            if action.get("value") not in ["low", "medium", "high", "critical"]:
                action["value"] = "medium"

        elif expected == "route":
            action["action_type"] = "route"
            if action.get("value") not in ["spam", "billing", "support", "sales", "engineering"]:
                action["value"] = "support"

    return action

# ---------------------------------------------------------------------------
# Run task
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_name: str, env_module: Any) -> None:
    from env.models import EmailTriageAction

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = env_module.EmailTriageEnv(task_name=task_name)
    obs_obj = env.reset()
    obs = obs_obj.model_dump()

    rewards: List[float] = []
    steps_taken = 0

    try:
        for step in range(1, MAX_STEPS_PER_TASK + 1):

            if obs.get("current_email") is None:
                break

            action_dict = get_action(client, task_name, obs)
            action_str  = json.dumps(action_dict)

            error_msg = None
            reward = 0.0
            done = False

            try:
                action = EmailTriageAction(**action_dict)
                obs_obj, reward, done, info = env.step(action)
                obs = obs_obj.model_dump()

            except Exception as exc:
                error_msg = str(exc)[:120]
                done = True

            rewards.append(reward)
            steps_taken = step

            log_step(step, action_str, reward, done, error_msg)

            if done or error_msg:
                break

        score = env.score()
        success = score >= SUCCESS_THRESHOLD

    finally:
        env.close()
        log_end(success, steps_taken, rewards)

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is required")

    import env.environment as env_module

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    tasks = ["task1_classify", "task2_prioritise", "task3_draft_reply"]

    for task in tasks:
        run_task(client, task, env_module)
        print()

if __name__ == "__main__":
    main()