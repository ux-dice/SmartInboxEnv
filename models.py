"""
models.py
---------
Core data models for SmartInboxEnv.

Written to be compatible with both:
  - pydantic v2  (if installed)
  - stdlib dataclasses  (fallback, zero dependencies)

OpenEnv validation requires:
  • Observation  has: task, step, email, prompt, done, info
  • Action types have the correct fields per task
  • Reward       has: value (float, 0.0-1.0), breakdown (dict), feedback (str)
  • StepResult   has: observation, reward, done, info
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# ── Try pydantic; fall back to stdlib dataclasses ──────────────────────────────
try:
    from pydantic import BaseModel, Field, field_validator
    _PYDANTIC = True
except ImportError:                         # pragma: no cover
    _PYDANTIC = False


# ── Enumerations ───────────────────────────────────────────────────────────────

class EmailCategory(str, Enum):
    SPAM   = "spam"
    URGENT = "urgent"
    NORMAL = "normal"


class Priority(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class ActionType(str, Enum):
    REPLY    = "reply"
    ESCALATE = "escalate"
    IGNORE   = "ignore"


class TaskName(str, Enum):
    EASY   = "classify"
    MEDIUM = "prioritize"
    HARD   = "triage"


# ── Model factory: produce Pydantic or dataclass versions ──────────────────────

if _PYDANTIC:
    # ── Pydantic v2 models ────────────────────────────────────────────────────

    class Email(BaseModel):
        id:      str
        subject: str
        body:    str
        sender:  str
        true_category:           EmailCategory
        true_priority:           Priority
        true_action:             ActionType
        ideal_response_keywords: List[str] = Field(default_factory=list)

        def public_dict(self) -> Dict[str, str]:
            return {"subject": self.subject, "body": self.body, "sender": self.sender}

    class Observation(BaseModel):
        task:   TaskName
        step:   int
        email:  Dict[str, str]
        prompt: str
        done:   bool = False
        info:   Dict[str, Any] = Field(default_factory=dict)

        def to_dict(self) -> Dict[str, Any]:
            return {
                "task":   self.task.value,
                "step":   self.step,
                "email":  self.email,
                "prompt": self.prompt,
                "done":   self.done,
                "info":   self.info,
            }

    class EasyAction(BaseModel):
        category: EmailCategory

        @field_validator("category", mode="before")
        @classmethod
        def _normalise(cls, v: Any) -> Any:
            return v.lower() if isinstance(v, str) else v

        def to_dict(self) -> Dict[str, Any]:
            return {"category": self.category.value}

    class MediumAction(BaseModel):
        priority: Priority

        @field_validator("priority", mode="before")
        @classmethod
        def _normalise(cls, v: Any) -> Any:
            return v.lower() if isinstance(v, str) else v

        def to_dict(self) -> Dict[str, Any]:
            return {"priority": self.priority.value}

    class HardAction(BaseModel):
        category: EmailCategory
        priority: Priority
        action:   ActionType
        response: str = ""

        @field_validator("category", "priority", "action", mode="before")
        @classmethod
        def _normalise(cls, v: Any) -> Any:
            return v.lower() if isinstance(v, str) else v

        def to_dict(self) -> Dict[str, Any]:
            return {
                "category": self.category.value,
                "priority": self.priority.value,
                "action":   self.action.value,
                "response": self.response,
            }

    class Reward(BaseModel):
        value:     float
        breakdown: Dict[str, float] = Field(default_factory=dict)
        feedback:  str = ""

        @field_validator("value")
        @classmethod
        def _clamp(cls, v: float) -> float:
            return round(max(0.0, min(1.0, v)), 6)

        def to_dict(self) -> Dict[str, Any]:
            return {"value": self.value, "breakdown": self.breakdown, "feedback": self.feedback}

    class StepResult(BaseModel):
        observation: Observation
        reward:      Reward
        done:        bool
        info:        Dict[str, Any] = Field(default_factory=dict)

else:
    # ── stdlib dataclasses fallback (zero dependencies) ───────────────────────

    def _clamp(v: float) -> float:
        return round(max(0.0, min(1.0, float(v))), 6)

    @dataclasses.dataclass
    class Email:                             # type: ignore[no-redef]
        id:      str
        subject: str
        body:    str
        sender:  str
        true_category:           EmailCategory = EmailCategory.NORMAL
        true_priority:           Priority       = Priority.LOW
        true_action:             ActionType     = ActionType.IGNORE
        ideal_response_keywords: List[str]      = dataclasses.field(default_factory=list)

        def public_dict(self) -> Dict[str, str]:
            return {"subject": self.subject, "body": self.body, "sender": self.sender}

    @dataclasses.dataclass
    class Observation:                       # type: ignore[no-redef]
        task:   TaskName
        step:   int
        email:  Dict[str, str]
        prompt: str
        done:   bool                  = False
        info:   Dict[str, Any]        = dataclasses.field(default_factory=dict)

        def to_dict(self) -> Dict[str, Any]:
            return {
                "task":   self.task.value,
                "step":   self.step,
                "email":  self.email,
                "prompt": self.prompt,
                "done":   self.done,
                "info":   self.info,
            }

    @dataclasses.dataclass
    class EasyAction:                        # type: ignore[no-redef]
        category: EmailCategory

        def to_dict(self) -> Dict[str, Any]:
            return {"category": self.category.value}

    @dataclasses.dataclass
    class MediumAction:                      # type: ignore[no-redef]
        priority: Priority

        def to_dict(self) -> Dict[str, Any]:
            return {"priority": self.priority.value}

    @dataclasses.dataclass
    class HardAction:                        # type: ignore[no-redef]
        category: EmailCategory
        priority: Priority
        action:   ActionType
        response: str = ""

        def to_dict(self) -> Dict[str, Any]:
            return {
                "category": self.category.value,
                "priority": self.priority.value,
                "action":   self.action.value,
                "response": self.response,
            }

    @dataclasses.dataclass
    class Reward:                            # type: ignore[no-redef]
        value:     float
        breakdown: Dict[str, float] = dataclasses.field(default_factory=dict)
        feedback:  str              = ""

        def __post_init__(self) -> None:
            self.value = _clamp(self.value)

        def to_dict(self) -> Dict[str, Any]:
            return {"value": self.value, "breakdown": self.breakdown, "feedback": self.feedback}

    @dataclasses.dataclass
    class StepResult:                        # type: ignore[no-redef]
        observation: Observation
        reward:      Reward
        done:        bool
        info:        Dict[str, Any] = dataclasses.field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_action_dict(task_name: str, d: Dict[str, Any]) -> Union[EasyAction, MediumAction, HardAction]:
    """
    Coerce a raw dict into the correct typed Action for the given task.
    Normalises enum values to lowercase before parsing.
    """
    # Normalise all string values
    normalised: Dict[str, Any] = {
        k: (v.lower().strip() if isinstance(v, str) else v)
        for k, v in d.items()
    }

    if task_name == TaskName.EASY.value:
        cat = normalised.get("category", "")
        return EasyAction(category=EmailCategory(cat))

    elif task_name == TaskName.MEDIUM.value:
        pri = normalised.get("priority", "")
        return MediumAction(priority=Priority(pri))

    elif task_name == TaskName.HARD.value:
        return HardAction(
            category=EmailCategory(normalised.get("category", "")),
            priority=Priority(normalised.get("priority", "")),
            action=ActionType(normalised.get("action", "")),
            response=str(normalised.get("response", "")),
        )
    else:
        raise ValueError(f"Unknown task name: '{task_name}'")
