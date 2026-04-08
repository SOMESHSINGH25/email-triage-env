from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    timestamp: str


class EmailTriageObservation(BaseModel):
    emails: List[Email] = Field(description="Emails currently in the inbox")
    current_email: Optional[Email] = Field(None, description="The email currently being acted on")
    step: int = Field(0, description="Current step number")
    task_description: str = Field("", description="Description of what the agent must accomplish")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task-specific context")


class EmailTriageAction(BaseModel):
    action_type: str = Field(
        description="One of: classify, prioritize, draft_reply, route, skip"
    )
    email_id: str = Field(description="ID of the email being acted on")
    value: str = Field(
        description=(
            "For classify: category name. "
            "For prioritize: urgency level (low/medium/high/critical). "
            "For draft_reply: the reply text. "
            "For route: queue name (support/billing/sales/engineering/spam). "
            "For skip: reason."
        )
    )


class EmailTriageReward(BaseModel):
    reward: float = Field(description="Reward for this step, in [0.0, 1.0]")
    breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Per-criterion reward breakdown"
    )
    feedback: str = Field("", description="Human-readable feedback on the action")