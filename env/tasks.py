"""
Task definitions for the Email Triage environment.

Task 1 — Classify (easy):
    Agent receives one email at a time and must classify it into the correct
    category. Reward is 1.0 for exact match, 0.5 for semantically adjacent
    categories, 0.0 otherwise.

Task 2 — Prioritise & Route (medium):
    Agent receives 6 emails and must assign urgency + route each one.
    Partial credit: urgency correct earns 0.4, routing correct earns 0.6 per email.

Task 3 — Draft Reply (hard):
    Agent receives an email that requires a reply and must draft a response
    that satisfies policy rules (apologise, give ETA, no blame), contains
    required keywords, and avoids forbidden phrases.
    Multi-component reward with partial credit throughout.
"""

from typing import Any, Dict, List, Optional, Tuple
from env.data import (
    GROUND_TRUTH, REFERENCE_REPLIES,
    VALID_CATEGORIES, VALID_URGENCIES, VALID_QUEUES,
    all_emails, get_emails_by_ids,
)
from env.models import Email, EmailTriageAction, EmailTriageObservation, EmailTriageReward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Semantically adjacent categories that deserve partial credit
_ADJACENT: Dict[str, str] = {
    "engineering": "support",
    "support": "engineering",
}


def _classify_reward(predicted: str, email_id: str) -> Tuple[float, str]:
    truth = GROUND_TRUTH[email_id]["category"]
    predicted = predicted.lower().strip()
    if predicted == truth:
        return 1.0, f"Correct! Category is '{truth}'."
    if _ADJACENT.get(predicted) == truth or _ADJACENT.get(truth) == predicted:
        return 0.5, f"Partially correct. '{predicted}' is adjacent to '{truth}'."
    return 0.0, f"Incorrect. Expected '{truth}', got '{predicted}'."


def _urgency_reward(predicted: str, email_id: str) -> Tuple[float, str]:
    truth = GROUND_TRUTH[email_id]["urgency"]
    predicted = predicted.lower().strip()
    if predicted not in VALID_URGENCIES:
        return 0.0, f"Invalid urgency '{predicted}'."
    if predicted == truth:
        return 1.0, "Correct urgency."
    # One level off → partial credit
    order = ["low", "medium", "high", "critical"]
    diff = abs(order.index(predicted) - order.index(truth))
    if diff == 1:
        return 0.5, f"Off by one level. Expected '{truth}'."
    return 0.0, f"Urgency too far off. Expected '{truth}', got '{predicted}'."


def _route_reward(predicted: str, email_id: str) -> Tuple[float, str]:
    truth = GROUND_TRUTH[email_id]["queue"]
    predicted = predicted.lower().strip()
    if predicted not in VALID_QUEUES:
        return 0.0, f"Invalid queue '{predicted}'."
    if predicted == truth:
        return 1.0, "Correct queue."
    return 0.0, f"Wrong queue. Expected '{truth}', got '{predicted}'."


def _reply_reward(reply: str, email_id: str) -> Tuple[float, Dict[str, float], str]:
    """Multi-component grader for draft reply task."""
    if email_id not in REFERENCE_REPLIES:
        return 0.0, {}, "No reference reply for this email."

    ref = REFERENCE_REPLIES[email_id]
    reply_lower = reply.lower()
    breakdown: Dict[str, float] = {}
    feedbacks: List[str] = []

    # 1. Length check (penalise empty or tiny replies)
    if len(reply.split()) < 10:
        return 0.0, {"length": 0.0}, "Reply too short (< 10 words). No credit awarded."

    # 2. Required keywords (0.4 total, evenly distributed)
    kw_score = 0.0
    kw_list = ref["required_keywords"]
    kw_hit = sum(1 for kw in kw_list if kw.lower() in reply_lower)
    kw_score = round(kw_hit / len(kw_list) * 0.4, 4)
    breakdown["keywords"] = kw_score
    feedbacks.append(f"Keywords matched: {kw_hit}/{len(kw_list)}")

    # 3. Forbidden phrase penalty (−0.1 each, floored at 0)
    forbidden_hits = [fp for fp in ref["forbidden_phrases"] if fp.lower() in reply_lower]
    penalty = min(len(forbidden_hits) * 0.1, 0.3)
    breakdown["forbidden_penalty"] = -penalty
    if forbidden_hits:
        feedbacks.append(f"Forbidden phrases found: {forbidden_hits}")

    # 4. Policy rules (0.4 total)
    policy = ref["policy_rules"]
    policy_score = 0.0
    per_rule = 0.4 / max(len(policy), 1)

    if policy.get("must_apologise"):
        if any(w in reply_lower for w in ["sorry", "apologise", "apologies", "apology"]):
            policy_score += per_rule
            feedbacks.append("Apologised: ✓")
        else:
            feedbacks.append("Apologised: ✗ (required)")

    if policy.get("must_give_eta"):
        if any(w in reply_lower for w in ["hours", "days", "minutes", "within", "business"]):
            policy_score += per_rule
            feedbacks.append("ETA given: ✓")
        else:
            feedbacks.append("ETA given: ✗ (required)")

    if policy.get("no_blame"):
        blame_words = ["your fault", "you should", "you didn't", "user error"]
        if not any(b in reply_lower for b in blame_words):
            policy_score += per_rule
            feedbacks.append("No blame: ✓")
        else:
            feedbacks.append("No blame: ✗ (policy violation)")

    breakdown["policy"] = round(policy_score, 4)

    # 5. Sycophancy penalty: if reply just agrees with everything / no actionable content
    action_words = ["will", "will process", "our team", "we'll", "investigating",
                    "refund", "looking into", "follow up", "resolve"]
    if not any(aw in reply_lower for aw in action_words):
        breakdown["sycophancy_penalty"] = -0.1
        feedbacks.append("No actionable commitment found (sycophancy penalty −0.1)")
    else:
        breakdown["sycophancy_penalty"] = 0.0

    total = sum(breakdown.values())
    total = round(max(0.0, min(1.0, total)), 4)
    return total, breakdown, " | ".join(feedbacks)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASK_EMAIL_IDS = {
    "task1_classify": ["e001", "e003", "e005", "e007", "e009", "e006"],
    "task2_prioritise": ["e003", "e004", "e005", "e006", "e009", "e010"],
    "task3_draft_reply": ["e005", "e004"],
}


class Task1Classify:
    name = "task1_classify"
    description = (
        "Email Classification (Easy): Classify each email into one of: "
        "spam, billing, support, sales, engineering. "
        "You must call action_type='classify' with the correct category."
    )

    def __init__(self):
        self.email_ids = TASK_EMAIL_IDS["task1_classify"]
        self.emails = get_emails_by_ids(self.email_ids)
        self.idx = 0
        self.rewards_per_step: List[float] = []

    def reset(self) -> EmailTriageObservation:
        self.idx = 0
        self.rewards_per_step = []
        return EmailTriageObservation(
            emails=self.emails,
            current_email=self.emails[0],
            step=0,
            task_description=self.description,
            context={"valid_categories": list(VALID_CATEGORIES)},
        )

    def step(self, action: EmailTriageAction) -> Tuple[EmailTriageObservation, float, bool, Dict]:
        email = self.emails[self.idx]

        if action.action_type != "classify":
            reward_val = 0.0
            feedback = f"Wrong action_type '{action.action_type}'. Expected 'classify'."
        elif action.email_id != email.id:
            reward_val = 0.0
            feedback = f"Wrong email_id. Expected '{email.id}'."
        else:
            reward_val, feedback = _classify_reward(action.value, email.id)

        self.rewards_per_step.append(reward_val)
        self.idx += 1
        done = self.idx >= len(self.emails)

        next_email = self.emails[self.idx] if not done else None
        obs = EmailTriageObservation(
            emails=self.emails,
            current_email=next_email,
            step=self.idx,
            task_description=self.description,
            context={"feedback": feedback, "valid_categories": list(VALID_CATEGORIES)},
        )
        info = {"feedback": feedback, "email_id": email.id}
        return obs, reward_val, done, info

    def score(self) -> float:
        if not self.rewards_per_step:
            return 0.0
        return round(sum(self.rewards_per_step) / len(self.rewards_per_step), 4)


class Task2Prioritise:
    name = "task2_prioritise"
    description = (
        "Email Prioritisation & Routing (Medium): For each email, first set urgency "
        "(action_type='prioritize', value=low|medium|high|critical), then route it "
        "(action_type='route', value=spam|billing|support|sales|engineering). "
        "You will alternate: prioritize → route → prioritize → route ..."
    )

    def __init__(self):
        self.email_ids = TASK_EMAIL_IDS["task2_prioritise"]
        self.emails = get_emails_by_ids(self.email_ids)
        self.email_idx = 0
        self.phase = "prioritize"  # alternates: prioritize → route
        self.rewards_per_step: List[float] = []
        self._urgency_reward_pending = 0.0

    def reset(self) -> EmailTriageObservation:
        self.email_idx = 0
        self.phase = "prioritize"
        self.rewards_per_step = []
        self._urgency_reward_pending = 0.0
        return self._make_obs(step=0, context={})

    def _make_obs(self, step: int, context: Dict) -> EmailTriageObservation:
        current = self.emails[self.email_idx] if self.email_idx < len(self.emails) else None
        return EmailTriageObservation(
            emails=self.emails,
            current_email=current,
            step=step,
            task_description=self.description,
            context={
                **context,
                "expected_action": self.phase,
                "valid_urgencies": list(VALID_URGENCIES),
                "valid_queues": list(VALID_QUEUES),
            },
        )

    def step(self, action: EmailTriageAction) -> Tuple[EmailTriageObservation, float, bool, Dict]:
        email = self.emails[self.email_idx]
        reward_val = 0.0
        feedback = ""

        if action.email_id != email.id:
            feedback = f"Wrong email_id. Expected '{email.id}'."
            self.rewards_per_step.append(0.0)
        elif self.phase == "prioritize":
            if action.action_type != "prioritize":
                feedback = f"Expected action_type='prioritize', got '{action.action_type}'."
            else:
                ur, feedback = _urgency_reward(action.value, email.id)
                # Urgency worth 40% of per-email score
                reward_val = round(ur * 0.4, 4)
                self._urgency_reward_pending = reward_val
            self.rewards_per_step.append(reward_val)
            self.phase = "route"
        else:  # route phase
            if action.action_type != "route":
                feedback = f"Expected action_type='route', got '{action.action_type}'."
            else:
                rr, feedback = _route_reward(action.value, email.id)
                # Route worth 60% of per-email score
                reward_val = round(rr * 0.6, 4)
            self.rewards_per_step.append(reward_val)
            self.phase = "prioritize"
            self.email_idx += 1

        step_num = len(self.rewards_per_step)
        done = self.email_idx >= len(self.emails)
        obs = self._make_obs(step=step_num, context={"feedback": feedback})
        info = {"feedback": feedback, "phase": self.phase, "email_id": email.id}
        return obs, reward_val, done, info

    def score(self) -> float:
        if not self.rewards_per_step:
            return 0.0
        # Max possible: 6 emails × (0.4 + 0.6) = 6.0 → normalise
        max_possible = len(self.emails) * 1.0
        return round(sum(self.rewards_per_step) / max_possible, 4)


class Task3DraftReply:
    name = "task3_draft_reply"
    description = (
        "Email Reply Drafting (Hard): Draft a professional reply to each email. "
        "Your reply must: apologise if appropriate, give a time estimate, "
        "not blame the customer, include relevant keywords, avoid forbidden phrases, "
        "and include actionable commitments. "
        "Use action_type='draft_reply', value=<your reply text>."
    )

    def __init__(self):
        self.email_ids = TASK_EMAIL_IDS["task3_draft_reply"]
        self.emails = get_emails_by_ids(self.email_ids)
        self.idx = 0
        self.rewards_per_step: List[float] = []

    def reset(self) -> EmailTriageObservation:
        self.idx = 0
        self.rewards_per_step = []
        return EmailTriageObservation(
            emails=self.emails,
            current_email=self.emails[0],
            step=0,
            task_description=self.description,
            context={
                "policy": (
                    "Apologise when appropriate. Give a time ETA. "
                    "Never blame the customer. Include actionable next steps."
                )
            },
        )

    def step(self, action: EmailTriageAction) -> Tuple[EmailTriageObservation, float, bool, Dict]:
        email = self.emails[self.idx]

        if action.action_type != "draft_reply":
            reward_val = 0.0
            breakdown: Dict[str, float] = {}
            feedback = f"Wrong action_type '{action.action_type}'. Expected 'draft_reply'."
        elif action.email_id != email.id:
            reward_val = 0.0
            breakdown = {}
            feedback = f"Wrong email_id. Expected '{email.id}'."
        else:
            reward_val, breakdown, feedback = _reply_reward(action.value, email.id)

        self.rewards_per_step.append(reward_val)
        self.idx += 1
        done = self.idx >= len(self.emails)

        next_email = self.emails[self.idx] if not done else None
        obs = EmailTriageObservation(
            emails=self.emails,
            current_email=next_email,
            step=self.idx,
            task_description=self.description,
            context={"feedback": feedback, "breakdown": breakdown},
        )
        info = {"feedback": feedback, "breakdown": breakdown, "email_id": email.id}
        return obs, reward_val, done, info

    def score(self) -> float:
        if not self.rewards_per_step:
            return 0.0
        return round(sum(self.rewards_per_step) / len(self.rewards_per_step), 4)


TASKS = {
    "task1_classify": Task1Classify,
    "task2_prioritise": Task2Prioritise,
    "task3_draft_reply": Task3DraftReply,
}