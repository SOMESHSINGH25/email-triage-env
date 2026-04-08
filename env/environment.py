"""
EmailTriageEnv — OpenEnv-compliant environment for email triage tasks.

Implements the standard OpenEnv interface:
    reset()  → EmailTriageObservation
    step()   → (EmailTriageObservation, float, bool, dict)
    state()  → dict
    close()  → None
"""

from typing import Any, Dict, Optional, Tuple

from env.models import EmailTriageAction, EmailTriageObservation, EmailTriageReward
from env.tasks import TASKS


class EmailTriageEnv:
    """
    Email Triage Environment.

    An AI agent learns to triage an inbox by:
        Task 1: Classifying emails into categories (easy)
        Task 2: Prioritising and routing emails (medium)
        Task 3: Drafting compliant replies (hard)
    """

    ENV_NAME = "email-triage"
    VERSION = "1.0.0"

    def __init__(self, task_name: str = "task1_classify"):
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}"
            )
        self.task_name = task_name
        self._task = TASKS[task_name]()
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._rewards_history: list = []
        self._last_obs: Optional[EmailTriageObservation] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> EmailTriageObservation:
        """Reset environment to initial state and return first observation."""
        self._task = TASKS[self.task_name]()
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._rewards_history = []
        self._last_obs = self._task.reset()
        return self._last_obs

    def step(
        self, action: EmailTriageAction
    ) -> Tuple[EmailTriageObservation, float, bool, Dict[str, Any]]:
        """
        Take one action in the environment.

        Returns:
            observation: New EmailTriageObservation
            reward:      Float in [0.0, 1.0]
            done:        True when episode is complete
            info:        Dict with feedback and debug info
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        obs, reward, done, info = self._task.step(action)

        # Penalise repeated no-ops / skip actions
        if action.action_type == "skip":
            reward = max(reward - 0.2, 0.0)
            info["skip_penalty"] = True

        self._step_count += 1
        self._cumulative_reward += reward
        self._rewards_history.append(reward)
        self._done = done
        self._last_obs = obs

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return current environment state (for checkpointing / debugging)."""
        return {
            "env_name": self.ENV_NAME,
            "version": self.VERSION,
            "task_name": self.task_name,
            "step_count": self._step_count,
            "done": self._done,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "rewards_history": [round(r, 4) for r in self._rewards_history],
            "score": self._task.score(),
            "current_observation": (
                self._last_obs.model_dump() if self._last_obs else None
            ),
        }

    def close(self) -> None:
        """Clean up resources (no-op for this environment)."""
        pass

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def score(self) -> float:
        """Final normalised score in [0.0, 1.0]."""
        return self._task.score()

    @staticmethod
    def available_tasks() -> list:
        return list(TASKS.keys())