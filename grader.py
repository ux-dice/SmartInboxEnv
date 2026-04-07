"""
grader.py
---------
Deterministic graders for all three SmartInboxEnv tasks.

All scores are in [0.0, 1.0]. Same inputs always yield same output.

EASY   (classify)   : binary 1.0 / 0.0
MEDIUM (prioritize) : partial 1.0 / 0.5 / 0.0  (off-by-one leniency)
HARD   (triage)     : shaped reward with strict dependency logic

Hard task reward design
-----------------------
Base weights:
  classification   ×0.30
  priority         ×0.20
  action           ×0.20
  response_quality ×0.30

Dependency penalties (applied after weighted sum):
  • Wrong classification  → multiply total by 0.60 (severe cascading error)
  • Wrong action          → cap response_score at 0.60 (wording can't rescue wrong decision)
  • VIP sender mishandled → additional 0.10 deduction if not escalated

Total is always clamped to [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, List

from models import ActionType, EasyAction, Email, HardAction, MediumAction, Priority, Reward
from tasks import is_vip_sender

# ── Priority ordering ──────────────────────────────────────────────────────────

_PRIORITY_RANK: dict = {
    Priority.LOW:    0,
    Priority.MEDIUM: 1,
    Priority.HIGH:   2,
}

# Hard task base weights — MUST sum to 1.0
_W_CATEGORY = 0.30
_W_PRIORITY  = 0.20
_W_ACTION    = 0.20
_W_RESPONSE  = 0.30


# ── Easy grader ───────────────────────────────────────────────────────────────

def grade_easy(action: EasyAction, email: Email) -> Reward:
    """Binary reward: 1.0 correct, 0.0 incorrect."""
    correct = action.category == email.true_category
    score   = 1.0 if correct else 0.0
    feedback = (
        f"Correct — '{action.category.value}' matches expected '{email.true_category.value}'."
        if correct else
        f"Incorrect — got '{action.category.value}', expected '{email.true_category.value}'."
    )
    return Reward(
        value=score,
        breakdown={"category_correct": score},
        feedback=feedback,
    )


# ── Medium grader ─────────────────────────────────────────────────────────────

def grade_medium(action: MediumAction, email: Email) -> Reward:
    """
    Partial-credit reward:
      1.0 → exact match
      0.5 → off by one level
      0.0 → opposite extreme (low vs high)
    """
    diff = abs(_PRIORITY_RANK[action.priority] - _PRIORITY_RANK[email.true_priority])
    if diff == 0:
        score, feedback = 1.0, f"Correct — '{action.priority.value}' is the right priority."
    elif diff == 1:
        score, feedback = 0.5, (
            f"Partial — got '{action.priority.value}', "
            f"expected '{email.true_priority.value}' (off by one level)."
        )
    else:
        score, feedback = 0.0, (
            f"Incorrect — got '{action.priority.value}', "
            f"expected '{email.true_priority.value}' (opposite extreme)."
        )
    return Reward(
        value=score,
        breakdown={"priority_score": score},
        feedback=feedback,
    )


# ── Hard grader helpers ───────────────────────────────────────────────────────

def _score_category(action: HardAction, email: Email) -> float:
    return 1.0 if action.category == email.true_category else 0.0


def _score_priority(action: HardAction, email: Email) -> float:
    diff = abs(_PRIORITY_RANK[action.priority] - _PRIORITY_RANK[email.true_priority])
    return 1.0 if diff == 0 else (0.5 if diff == 1 else 0.0)


def _score_action(action: HardAction, email: Email) -> float:
    return 1.0 if action.action == email.true_action else 0.0


def _score_response(action: HardAction, email: Email) -> float:
    """
    Keyword-coverage scoring — deterministic, zero randomness.

    • ignore action expected → full marks for empty response; penalise any reply.
    • no keywords defined    → binary: non-empty response = 1.0.
    • otherwise              → fraction of ideal keywords present.
    • wrong action chosen    → cap at 0.60 (action mistake limits response value).
    """
    response   = action.response.strip()
    ideal_keys: List[str] = email.ideal_response_keywords

    if email.true_action == ActionType.IGNORE:
        # Spam/newsletters: correct behaviour is silence
        return 1.0 if response == "" else 0.2

    if not ideal_keys:
        # No keywords specified — reward a substantive reply
        return 1.0 if len(response) > 20 else 0.0

    hits = sum(1 for kw in ideal_keys if kw.lower() in response.lower())
    raw  = hits / len(ideal_keys)

    # Cap response score if the agent chose the wrong action
    if action.action != email.true_action:
        raw = min(raw, 0.60)

    return round(raw, 6)


# ── Hard grader ───────────────────────────────────────────────────────────────

def grade_hard(action: HardAction, email: Email) -> Reward:
    """
    Shaped reward with dependency penalties.

    Base: category×0.30 + priority×0.20 + action×0.20 + response×0.30

    Penalties applied multiplicatively / additively:
      • Wrong classification → ×0.60 multiplier on total (cascading error)
      • VIP sender not escalated → −0.10 deduction
    """
    cat_score  = _score_category(action, email)
    pri_score  = _score_priority(action, email)
    act_score  = _score_action(action, email)
    resp_score = _score_response(action, email)

    # Weighted base score
    base = (
        cat_score  * _W_CATEGORY +
        pri_score  * _W_PRIORITY  +
        act_score  * _W_ACTION    +
        resp_score * _W_RESPONSE
    )

    penalty_notes: list = []

    # Penalty 1: wrong classification cascades badly — it means the agent
    # misunderstood the nature of the email entirely.
    if cat_score < 1.0:
        base *= 0.60
        penalty_notes.append("classification wrong — 0.60x multiplier applied")

    # Penalty 2: VIP sender (CEO/VP/Director) must be escalated.
    vip_penalty = 0.0
    if is_vip_sender(email.sender) and action.action != ActionType.ESCALATE:
        vip_penalty = 0.10
        penalty_notes.append("VIP sender not escalated — 0.10 deduction")

    total = round(max(0.0, min(1.0, base - vip_penalty)), 4)

    breakdown = {
        f"category  (x{_W_CATEGORY})":  round(cat_score  * _W_CATEGORY, 4),
        f"priority  (x{_W_PRIORITY})":   round(pri_score  * _W_PRIORITY,  4),
        f"action    (x{_W_ACTION})":     round(act_score  * _W_ACTION,    4),
        f"response  (x{_W_RESPONSE})":   round(resp_score * _W_RESPONSE,  4),
        "penalties":                     round(-(vip_penalty + (base - base * 0.60 if cat_score < 1.0 else 0)), 4),
        "total":                         total,
    }

    issues: list = []
    if cat_score  < 1.0: issues.append(f"category mismatch (expected {email.true_category.value})")
    if pri_score  < 1.0: issues.append(f"priority off (expected {email.true_priority.value})")
    if act_score  < 1.0: issues.append(f"action mismatch (expected {email.true_action.value})")
    if resp_score < 1.0: issues.append(f"response missing key concepts (score={resp_score:.2f})")
    issues.extend(penalty_notes)

    feedback = (
        "All four dimensions correct!" if not issues
        else "Issues: " + "; ".join(issues) + "."
    )

    return Reward(value=total, breakdown=breakdown, feedback=feedback)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def grade(task_name: str, action: Any, email: Email) -> Reward:
    """Route to the correct grader. Raises ValueError for unknown tasks."""
    if task_name == "classify":
        if not isinstance(action, EasyAction):
            raise TypeError(f"Expected EasyAction for 'classify', got {type(action).__name__}")
        return grade_easy(action, email)
    elif task_name == "prioritize":
        if not isinstance(action, MediumAction):
            raise TypeError(f"Expected MediumAction for 'prioritize', got {type(action).__name__}")
        return grade_medium(action, email)
    elif task_name == "triage":
        if not isinstance(action, HardAction):
            raise TypeError(f"Expected HardAction for 'triage', got {type(action).__name__}")
        return grade_hard(action, email)
    else:
        raise ValueError(f"Unknown task '{task_name}'. Valid: classify, prioritize, triage")
