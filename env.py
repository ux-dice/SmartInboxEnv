"""
env.py
------
SmartInboxEnv — OpenEnv-compatible AI email triage environment.

OpenEnv contract (strictly implemented):
  reset(task, seed)  → Observation
  step(action_dict)  → (Observation, Reward, done: bool, info: dict)
  state()            → dict

All public methods are type-annotated.
No global mutable state — each instance is independent.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from grader import grade
from models import (
    HardAction, Observation, Reward, StepResult, TaskName,
    parse_action_dict,
)
from tasks import ALL_TASKS, MAX_STEPS, build_prompt, sample_email


class SmartInboxEnv:
    """
    OpenEnv environment for AI email triage.

    Tasks
    -----
    classify   (easy)   — classify an email as spam/urgent/normal
    prioritize (medium) — assign low/medium/high priority
    triage     (hard)   — classify + prioritize + action + response

    Usage
    -----
    >>> env = SmartInboxEnv()
    >>> obs = env.reset("triage", seed=42)
    >>> obs_next, reward, done, info = env.step({
    ...     "category": "urgent",
    ...     "priority": "high",
    ...     "action":   "escalate",
    ...     "response": "Escalating to on-call immediately.",
    ... })
    >>> print(reward.value)
    """

    # ── Class-level metadata (OpenEnv spec) ───────────────────────────────────
    ENV_NAME    : str = "SmartInboxEnv"
    ENV_VERSION : str = "1.0.0"
    DESCRIPTION : str = "AI email triage and decision-making benchmark environment."

    def __init__(self) -> None:
        # All mutable state lives here; never at class level
        self._task_name  : Optional[str]     = None
        self._email      : Optional[Any]     = None   # models.Email
        self._step_count : int               = 0
        self._done       : bool              = False
        self._seed       : Optional[int]     = None
        self._history    : list[dict]        = []
        self._cumulative_reward : float      = 0.0

    # ── OpenEnv required interface ────────────────────────────────────────────

    def reset(
        self,
        task: str = "classify",
        seed: Optional[int] = None,
    ) -> Observation:
        """
        Initialise (or re-initialise) the environment for a given task.

        Parameters
        ----------
        task : str
            One of 'classify', 'prioritize', 'triage'.
        seed : int, optional
            Random seed for reproducible email selection.

        Returns
        -------
        Observation
            The initial observation containing the email and prompt.

        Raises
        ------
        ValueError
            If task name is not recognised.
        """
        if task not in ALL_TASKS:
            raise ValueError(
                f"Unknown task '{task}'. "
                f"Valid tasks: {ALL_TASKS}"
            )

        self._task_name          = task
        self._seed               = seed
        self._email              = sample_email(task, seed=seed)
        self._step_count         = 0
        self._done               = False
        self._history            = []
        self._cumulative_reward  = 0.0

        return self._build_observation(done=False)

    def step(
        self,
        action_dict: Dict[str, Any],
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Parameters
        ----------
        action_dict : dict
            Raw action dictionary.  Keys depend on task:
              classify   → {"category": str}
              prioritize → {"priority": str}
              triage     → {"category": str, "priority": str,
                            "action": str, "response": str}

        Returns
        -------
        (Observation, Reward, done, info)
            OpenEnv standard 4-tuple.

        Raises
        ------
        RuntimeError
            If step() is called before reset() or after episode ends.
        """
        self._assert_ready()

        # ── Parse action (returns zero-reward on bad input, does NOT crash) ──
        parse_error: Optional[str] = None
        action: Any = None

        try:
            action = parse_action_dict(self._task_name, action_dict)
        except Exception as exc:
            parse_error = str(exc)

        if parse_error is not None:
            reward = Reward(
                value=0.0,
                breakdown={"parse_error": 0.0},
                feedback=f"Action parse failed: {parse_error}",
            )
            self._done = True          # terminal on bad action
            obs  = self._build_observation(done=True)
            info = {
                "error":     parse_error,
                "step":      self._step_count,
                "max_steps": MAX_STEPS[self._task_name],
            }
            self._record_history(action_dict, reward)
            return obs, reward, self._done, info

        # ── Grade ─────────────────────────────────────────────────────────────
        reward = grade(self._task_name, action, self._email)

        # ── Advance ───────────────────────────────────────────────────────────
        self._step_count        += 1
        self._cumulative_reward += reward.value
        self._done               = self._step_count >= MAX_STEPS[self._task_name]

        self._record_history(action_dict, reward)

        obs  = self._build_observation(done=self._done)
        info = self._build_info(reward)

        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """
        Return the complete current environment state.

        Suitable for serialisation (all values are JSON-serialisable).
        """
        return {
            "env":               self.ENV_NAME,
            "version":           self.ENV_VERSION,
            "task":              self._task_name,
            "step":              self._step_count,
            "done":              self._done,
            "seed":              self._seed,
            "email_id":          self._email.id if self._email else None,
            "cumulative_reward": round(self._cumulative_reward, 6),
            "history":           copy.deepcopy(self._history),
            "max_steps":         MAX_STEPS.get(self._task_name, 0) if self._task_name else 0,
        }

    # ── Convenience ───────────────────────────────────────────────────────────

    def render(self) -> str:
        """Return a human-readable snapshot of the current state."""
        if self._task_name is None:
            return "[SmartInboxEnv] Not initialised — call reset() first."

        lines = [
            f"[SmartInboxEnv] task={self._task_name}  step={self._step_count}  done={self._done}",
            f"  email_id : {self._email.id}",
            f"  subject  : {self._email.subject}",
            f"  steps    : {self._step_count}/{MAX_STEPS[self._task_name]}",
            f"  cumulative_reward: {self._cumulative_reward:.4f}",
        ]
        if self._history:
            last = self._history[-1]
            lines.append(f"  last_reward : {last['reward']:.4f}")
            lines.append(f"  feedback    : {last['feedback']}")
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assert_ready(self) -> None:
        """Raise RuntimeError if the environment is not in a valid state for step()."""
        if self._task_name is None:
            raise RuntimeError(
                "SmartInboxEnv: step() called before reset(). "
                "Call env.reset(task='classify') first."
            )
        if self._done:
            raise RuntimeError(
                "SmartInboxEnv: step() called on a completed episode. "
                "Call env.reset() to start a new episode."
            )

    def _build_observation(self, done: bool) -> Observation:
        """Construct the current Observation (public email fields only)."""
        prompt = build_prompt(self._task_name, self._email)
        return Observation(
            task=TaskName(self._task_name),
            step=self._step_count,
            email=self._email.public_dict(),
            prompt=prompt,
            done=done,
            info={
                "email_id": self._email.id,
                "task":     self._task_name,
            },
        )

    def _build_info(self, reward: Reward) -> Dict[str, Any]:
        """Build the info dict returned by step()."""
        return {
            "feedback":          reward.feedback,
            "breakdown":         reward.breakdown,
            "step":              self._step_count,
            "max_steps":         MAX_STEPS[self._task_name],
            "cumulative_reward": round(self._cumulative_reward, 6),
            "error":             None,
        }

    def _record_history(self, action_dict: Dict[str, Any], reward: Reward) -> None:
        """Append a step record to internal history."""
        self._history.append({
            "step":     self._step_count,
            "action":   copy.deepcopy(action_dict),
            "reward":   round(reward.value, 6),
            "feedback": reward.feedback,
        })
