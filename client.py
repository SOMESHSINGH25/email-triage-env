"""
client.py — HTTP client for the Email Triage OpenEnv environment.

Wraps the FastAPI server defined in api/main.py, which exposes:
    POST /reset   → initial observation
    POST /step    → next observation + reward + done + info
    GET  /state   → current env state
    GET  /tasks   → list available tasks
    GET  /health  → health check

Default base URL: http://localhost:7860  (HF Spaces port).
Override with ENV_BASE_URL environment variable.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

DEFAULT_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")


class EnvClient:
    """
    HTTP client for the Email Triage environment server.

    Example
    -------
    >>> client = EnvClient()
    >>> client.wait_until_healthy()
    >>> obs = client.reset("task1_classify")["observation"]
    >>> result = client.step("classify", "e001", "spam")
    >>> print(result["reward"], result["done"])
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Informational endpoints
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """
        GET /health → {"status": "ok", "env": "email-triage", "version": "1.0.0"}
        """
        resp = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> list[str]:
        """
        GET /tasks → ["task1_classify", "task2_prioritise", "task3_draft_reply"]
        """
        resp = self._session.get(f"{self.base_url}/tasks", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("tasks", [])

    def state(self) -> dict[str, Any]:
        """GET /state — current env state (step count, cumulative reward, history …)."""
        resp = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self, task_name: str = "task1_classify") -> dict[str, Any]:
        """
        POST /reset

        Returns the server response:
            {"observation": {...}, "done": False}

        Parameters
        ----------
        task_name:
            "task1_classify" | "task2_prioritise" | "task3_draft_reply"
        """
        resp = self._session.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def step(
        self,
        action_type: str,
        email_id: str,
        value: str,
    ) -> dict[str, Any]:
        """
        POST /step

        Returns:
            {
                "observation": {...},
                "reward":      float,
                "done":        bool,
                "info":        {"feedback": str, ...}
            }

        Parameters
        ----------
        action_type:
            "classify" | "prioritize" | "route" | "draft_reply" | "skip"
        email_id:
            ID of the email being acted on, e.g. "e001".
        value:
            Predicted category / urgency / queue, or free-text reply body.
        """
        resp = self._session.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "email_id": email_id, "value": value},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def wait_until_healthy(self, retries: int = 15, delay: float = 2.0) -> None:
        """
        Poll /health until {"status": "ok"} is returned, or raise RuntimeError.

        Useful in Docker / HF Spaces where the server may take a few seconds
        to initialise.
        """
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                data = self.health()
                if data.get("status") in ("ok", "healthy"):
                    return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
            if attempt < retries:
                time.sleep(delay)

        raise RuntimeError(
            f"Server at {self.base_url} did not become healthy after "
            f"{retries} attempts."
            + (f" Last error: {last_exc}" if last_exc else "")
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"EnvClient(base_url={self.base_url!r})"