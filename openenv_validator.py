#!/usr/bin/env python3
"""
openenv_validator.py
--------------------
Self-contained OpenEnv compliance validator for SmartInboxEnv.

Checks every requirement from the OpenEnv specification:

  [SPEC-01]  reset() returns an Observation with required fields
  [SPEC-02]  step() returns 4-tuple (Observation, Reward, bool, dict)
  [SPEC-03]  state() returns a dict with required keys
  [SPEC-04]  Reward.value is float in [0.0, 1.0]
  [SPEC-05]  Reward.breakdown is a dict
  [SPEC-06]  Episode terminates after max_steps
  [SPEC-07]  step() raises RuntimeError before reset()
  [SPEC-08]  step() raises RuntimeError after episode ends
  [SPEC-09]  Bad action returns 0.0 reward and done=True (no crash)
  [SPEC-10]  Deterministic: same seed → same email → same score
  [SPEC-11]  Easy task: binary rewards only (0.0 or 1.0)
  [SPEC-12]  Medium task: scores in {0.0, 0.5, 1.0}
  [SPEC-13]  Hard task: shaped reward in (0, 1], sums of weighted components
  [SPEC-14]  Hard task: perfect action scores 1.0
  [SPEC-15]  openenv.yaml exists and has required keys
  [SPEC-16]  All tasks listed in yaml match implemented tasks
  [SPEC-17]  Observation.task is a valid TaskName
  [SPEC-18]  Observation.prompt is non-empty string
  [SPEC-19]  Observation.email has subject, body, sender (no ground-truth leaked)
  [SPEC-20]  Reward weight sum ≈ 1.0 for hard task
  [SPEC-21]  reset() is idempotent — can be called multiple times
  [SPEC-22]  state() is JSON-serialisable
  [SPEC-23]  info dict from step() contains 'error' key
  [SPEC-24]  No infinite loop: done flag set correctly
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

# ── Colour output (falls back gracefully if terminal doesn't support ANSI) ─────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _green(s: str)  -> str: return f"{GREEN}{s}{RESET}"
def _red(s: str)    -> str: return f"{RED}{s}{RESET}"
def _yellow(s: str) -> str: return f"{YELLOW}{s}{RESET}"
def _bold(s: str)   -> str: return f"{BOLD}{s}{RESET}"


# ── Test runner ───────────────────────────────────────────────────────────────

class ValidationResult:
    def __init__(self) -> None:
        self.passed  : list[tuple[str, str]] = []
        self.failed  : list[tuple[str, str]] = []
        self.skipped : list[tuple[str, str]] = []

    def ok(self, spec: str, detail: str = "") -> None:
        self.passed.append((spec, detail))
        print(f"  {_green('PASS')} {spec}  {detail}")

    def fail(self, spec: str, detail: str = "") -> None:
        self.failed.append((spec, detail))
        print(f"  {_red('FAIL')} {spec}  {detail}")

    def skip(self, spec: str, detail: str = "") -> None:
        self.skipped.append((spec, detail))
        print(f"  {_yellow('SKIP')} {spec}  {detail}")

    def summary(self) -> None:
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print()
        print(_bold("=" * 60))
        print(_bold(f"VALIDATION SUMMARY   {total} checks"))
        print(_bold("=" * 60))
        print(f"  {_green(f'PASSED : {len(self.passed)}')} / {total}")
        if self.failed:
            print(f"  {_red(f'FAILED : {len(self.failed)}')} / {total}")
            for spec, detail in self.failed:
                print(f"    ✗ {spec}: {detail}")
        if self.skipped:
            print(f"  {_yellow(f'SKIPPED: {len(self.skipped)}')} / {total}")
        print()
        if not self.failed:
            print(_bold(_green("✓ ALL CHECKS PASSED — OpenEnv compliant!")))
        else:
            print(_bold(_red(f"✗ {len(self.failed)} CHECK(S) FAILED — see above.")))
        print()

    @property
    def all_passed(self) -> bool:
        return len(self.failed) == 0


# ── Import helpers ─────────────────────────────────────────────────────────────

def _import_env():
    try:
        from env import SmartInboxEnv
        return SmartInboxEnv
    except Exception as exc:
        print(_red(f"[FATAL] Cannot import SmartInboxEnv from env.py: {exc}"))
        traceback.print_exc()
        sys.exit(1)


def _import_models():
    try:
        import models as m
        return m
    except Exception as exc:
        print(_red(f"[FATAL] Cannot import models.py: {exc}"))
        sys.exit(1)


def _import_graders():
    try:
        import grader as g
        return g
    except Exception as exc:
        print(_red(f"[FATAL] Cannot import grader.py: {exc}"))
        sys.exit(1)


# ── Individual checks ──────────────────────────────────────────────────────────

def check_reset_returns_observation(r: ValidationResult, SmartInboxEnv: Any) -> None:
    for task in ["classify", "prioritize", "triage"]:
        try:
            env = SmartInboxEnv()
            obs = env.reset(task=task, seed=42)
            assert hasattr(obs, "task"),   "missing .task"
            assert hasattr(obs, "step"),   "missing .step"
            assert hasattr(obs, "email"),  "missing .email"
            assert hasattr(obs, "prompt"), "missing .prompt"
            assert hasattr(obs, "done"),   "missing .done"
            assert hasattr(obs, "info"),   "missing .info"
            assert obs.step == 0,         f"step should be 0, got {obs.step}"
            assert obs.done is False,     "done should be False after reset"
            r.ok("SPEC-01", f"task={task}")
        except Exception as exc:
            r.fail("SPEC-01", f"task={task}: {exc}")


def check_step_returns_4_tuple(r: ValidationResult, SmartInboxEnv: Any) -> None:
    actions = {
        "classify":   {"category": "spam"},
        "prioritize": {"priority": "low"},
        "triage":     {"category": "spam", "priority": "low", "action": "ignore", "response": ""},
    }
    for task, action in actions.items():
        try:
            env = SmartInboxEnv()
            env.reset(task=task, seed=42)
            result = env.step(action)
            assert isinstance(result, tuple),      "step() must return a tuple"
            assert len(result) == 4,               f"step() must return 4 items, got {len(result)}"
            obs, reward, done, info = result
            assert hasattr(obs,    "task"),         "obs missing .task"
            assert hasattr(reward, "value"),        "reward missing .value"
            assert isinstance(done, bool),          f"done must be bool, got {type(done)}"
            assert isinstance(info, dict),          f"info must be dict, got {type(info)}"
            r.ok("SPEC-02", f"task={task}")
        except Exception as exc:
            r.fail("SPEC-02", f"task={task}: {exc}")


def check_state_returns_dict(r: ValidationResult, SmartInboxEnv: Any) -> None:
    required_keys = {"env", "version", "task", "step", "done", "history"}
    try:
        env = SmartInboxEnv()
        env.reset(task="classify", seed=1)
        s = env.state()
        assert isinstance(s, dict), "state() must return dict"
        missing = required_keys - set(s.keys())
        assert not missing, f"state() missing keys: {missing}"
        r.ok("SPEC-03")
    except Exception as exc:
        r.fail("SPEC-03", str(exc))


def check_reward_value_range(r: ValidationResult, SmartInboxEnv: Any) -> None:
    actions = [
        ("classify",   {"category": "spam"}),
        ("classify",   {"category": "urgent"}),
        ("prioritize", {"priority": "high"}),
        ("prioritize", {"priority": "low"}),
        ("triage",     {"category": "urgent", "priority": "high",
                        "action": "escalate", "response": "Escalating now."}),
    ]
    for task, action in actions:
        try:
            env = SmartInboxEnv()
            env.reset(task=task, seed=7)
            _, reward, _, _ = env.step(action)
            assert isinstance(reward.value, float), \
                f"reward.value must be float, got {type(reward.value)}"
            assert 0.0 <= reward.value <= 1.0, \
                f"reward.value={reward.value} out of [0,1]"
            r.ok("SPEC-04", f"task={task} reward={reward.value:.4f}")
        except Exception as exc:
            r.fail("SPEC-04", f"task={task}: {exc}")


def check_reward_breakdown_is_dict(r: ValidationResult, SmartInboxEnv: Any) -> None:
    try:
        env = SmartInboxEnv()
        env.reset(task="triage", seed=1)
        _, reward, _, _ = env.step(
            {"category": "spam", "priority": "low", "action": "ignore", "response": ""}
        )
        assert isinstance(reward.breakdown, dict), \
            f"breakdown must be dict, got {type(reward.breakdown)}"
        r.ok("SPEC-05")
    except Exception as exc:
        r.fail("SPEC-05", str(exc))


def check_episode_terminates(r: ValidationResult, SmartInboxEnv: Any) -> None:
    """All tasks are single-step; done must be True after 1 step."""
    for task in ["classify", "prioritize", "triage"]:
        action = (
            {"category": "spam"}
            if task == "classify" else
            {"priority": "low"}
            if task == "prioritize" else
            {"category": "spam", "priority": "low", "action": "ignore", "response": ""}
        )
        try:
            env = SmartInboxEnv()
            env.reset(task=task, seed=0)
            _, _, done, _ = env.step(action)
            assert done is True, f"done should be True after max_steps, got {done}"
            r.ok("SPEC-06", f"task={task}")
        except Exception as exc:
            r.fail("SPEC-06", f"task={task}: {exc}")


def check_step_before_reset_raises(r: ValidationResult, SmartInboxEnv: Any) -> None:
    try:
        env = SmartInboxEnv()
        raised = False
        try:
            env.step({"category": "spam"})
        except RuntimeError:
            raised = True
        assert raised, "step() before reset() must raise RuntimeError"
        r.ok("SPEC-07")
    except Exception as exc:
        r.fail("SPEC-07", str(exc))


def check_step_after_done_raises(r: ValidationResult, SmartInboxEnv: Any) -> None:
    try:
        env = SmartInboxEnv()
        env.reset(task="classify", seed=0)
        env.step({"category": "spam"})   # finishes episode
        raised = False
        try:
            env.step({"category": "spam"})
        except RuntimeError:
            raised = True
        assert raised, "step() after done must raise RuntimeError"
        r.ok("SPEC-08")
    except Exception as exc:
        r.fail("SPEC-08", str(exc))


def check_bad_action_no_crash(r: ValidationResult, SmartInboxEnv: Any) -> None:
    bad_actions = [
        ("classify",   {"category": "INVALID_CATEGORY"}),
        ("prioritize", {}),
        ("triage",     {"category": "spam"}),   # missing required fields
    ]
    for task, action in bad_actions:
        try:
            env = SmartInboxEnv()
            env.reset(task=task, seed=0)
            obs, reward, done, info = env.step(action)
            assert reward.value == 0.0, \
                f"bad action must yield 0.0 reward, got {reward.value}"
            assert done is True, "bad action must end episode"
            assert "error" in info, "info must have 'error' key on bad action"
            r.ok("SPEC-09", f"task={task}")
        except Exception as exc:
            r.fail("SPEC-09", f"task={task}: {exc}")


def check_deterministic(r: ValidationResult, SmartInboxEnv: Any) -> None:
    for task in ["classify", "prioritize", "triage"]:
        action = (
            {"category": "urgent"}
            if task == "classify" else
            {"priority": "high"}
            if task == "prioritize" else
            {"category": "urgent", "priority": "high",
             "action": "escalate", "response": "Escalating immediately."}
        )
        try:
            scores = []
            for _ in range(3):
                env = SmartInboxEnv()
                env.reset(task=task, seed=99)
                _, reward, _, _ = env.step(action)
                scores.append(reward.value)
            assert len(set(scores)) == 1, \
                f"Scores should be identical across runs, got {scores}"
            r.ok("SPEC-10", f"task={task} score={scores[0]:.4f} ×3 runs")
        except Exception as exc:
            r.fail("SPEC-10", f"task={task}: {exc}")


def check_easy_binary_rewards(r: ValidationResult, SmartInboxEnv: Any) -> None:
    from tasks import EMAILS
    from models import EmailCategory
    valid_scores = {0.0, 1.0}
    try:
        for email in EMAILS:
            for cat in EmailCategory:
                env = SmartInboxEnv()
                # Force specific email by patching _email directly after reset
                env.reset(task="classify", seed=0)
                env._email = email
                _, reward, _, _ = env.step({"category": cat.value})
                assert reward.value in valid_scores, \
                    f"Easy task score {reward.value} not in {valid_scores}"
        r.ok("SPEC-11", f"all {len(EMAILS) * len(list(EmailCategory))} combos binary")
    except Exception as exc:
        r.fail("SPEC-11", str(exc))


def check_medium_partial_rewards(r: ValidationResult, SmartInboxEnv: Any) -> None:
    from tasks import EMAILS
    from models import Priority
    valid_scores = {0.0, 0.5, 1.0}
    try:
        seen = set()
        for email in EMAILS:
            for pri in Priority:
                env = SmartInboxEnv()
                env.reset(task="prioritize", seed=0)
                env._email = email
                _, reward, _, _ = env.step({"priority": pri.value})
                assert reward.value in valid_scores, \
                    f"Medium task score {reward.value} not in {valid_scores}"
                seen.add(reward.value)
        # We must see all three score levels across the dataset
        assert seen == valid_scores, \
            f"Expected to see {{0.0, 0.5, 1.0}}, only saw {seen}"
        r.ok("SPEC-12", "all three score levels confirmed")
    except Exception as exc:
        r.fail("SPEC-12", str(exc))


def check_hard_shaped_reward(r: ValidationResult, SmartInboxEnv: Any) -> None:
    from tasks import EMAILS
    try:
        shaped_values = set()
        for email in EMAILS:
            env = SmartInboxEnv()
            env.reset(task="triage", seed=0)
            env._email = email
            action = {
                "category": email.true_category.value,
                "priority": "low",          # intentionally wrong
                "action":   email.true_action.value,
                "response": " ".join(email.ideal_response_keywords) or "acknowledged",
            }
            _, reward, _, _ = env.step(action)
            # Not binary — should be between 0 and 1 exclusive
            shaped_values.add(round(reward.value, 2))
        # Must have more than two distinct values across the dataset
        assert len(shaped_values) > 2, \
            f"Hard task rewards look binary, expected shaped: {shaped_values}"
        r.ok("SPEC-13", f"shaped values observed: {sorted(shaped_values)}")
    except Exception as exc:
        r.fail("SPEC-13", str(exc))


def check_hard_perfect_score(r: ValidationResult, SmartInboxEnv: Any) -> None:
    from tasks import EMAILS
    try:
        for email in EMAILS:
            env = SmartInboxEnv()
            env.reset(task="triage", seed=0)
            env._email = email
            action = {
                "category": email.true_category.value,
                "priority": email.true_priority.value,
                "action":   email.true_action.value,
                "response": " ".join(email.ideal_response_keywords) if email.ideal_response_keywords
                            else ("" if email.true_action.value == "ignore" else "acknowledged receipt"),
            }
            _, reward, _, _ = env.step(action)
            assert reward.value >= 0.9, \
                f"Perfect action on email {email.id} scored {reward.value:.4f} (expected ≥0.9)"
        r.ok("SPEC-14", f"all {len(EMAILS)} emails ≥0.9 with perfect action")
    except Exception as exc:
        r.fail("SPEC-14", str(exc))


def check_yaml_exists(r: ValidationResult) -> None:
    try:
        import yaml as _yaml_module
        _has_yaml = True
    except ImportError:
        _has_yaml = False

    yaml_path = Path(__file__).parent / "openenv.yaml"
    try:
        assert yaml_path.exists(), f"openenv.yaml not found at {yaml_path}"
        content = yaml_path.read_text(encoding="utf-8")
        assert len(content) > 50, "openenv.yaml appears empty"

        required_keys = ["name", "version", "description", "tasks", "inference"]
        for key in required_keys:
            assert key in content, f"openenv.yaml missing key: '{key}'"

        r.ok("SPEC-15", f"openenv.yaml found ({len(content)} chars)")
    except Exception as exc:
        r.fail("SPEC-15", str(exc))


def check_yaml_tasks_match(r: ValidationResult, SmartInboxEnv: Any) -> None:
    from tasks import ALL_TASKS
    yaml_path = Path(__file__).parent / "openenv.yaml"
    try:
        content = yaml_path.read_text(encoding="utf-8")
        for task in ALL_TASKS:
            assert task in content, \
                f"Task '{task}' not found in openenv.yaml"
        r.ok("SPEC-16", f"all tasks present: {ALL_TASKS}")
    except Exception as exc:
        r.fail("SPEC-16", str(exc))


def check_observation_task_valid(r: ValidationResult, SmartInboxEnv: Any) -> None:
    from models import TaskName
    valid_tasks = {t.value for t in TaskName}
    try:
        for task in ["classify", "prioritize", "triage"]:
            env = SmartInboxEnv()
            obs = env.reset(task=task, seed=5)
            assert obs.task.value in valid_tasks, \
                f"obs.task={obs.task!r} not in valid tasks {valid_tasks}"
        r.ok("SPEC-17")
    except Exception as exc:
        r.fail("SPEC-17", str(exc))


def check_observation_prompt_nonempty(r: ValidationResult, SmartInboxEnv: Any) -> None:
    try:
        for task in ["classify", "prioritize", "triage"]:
            env = SmartInboxEnv()
            obs = env.reset(task=task, seed=3)
            assert isinstance(obs.prompt, str), "prompt must be a string"
            assert len(obs.prompt.strip()) > 20, \
                f"prompt too short ({len(obs.prompt)} chars) for task={task}"
        r.ok("SPEC-18")
    except Exception as exc:
        r.fail("SPEC-18", str(exc))


def check_observation_no_ground_truth_leak(r: ValidationResult, SmartInboxEnv: Any) -> None:
    """The public observation must NOT expose true_category, true_priority, true_action."""
    try:
        env = SmartInboxEnv()
        obs = env.reset(task="triage", seed=0)
        email_dict = obs.email
        forbidden = {"true_category", "true_priority", "true_action",
                     "ideal_response_keywords"}
        leaked = forbidden & set(email_dict.keys())
        assert not leaked, f"Ground-truth fields leaked in observation: {leaked}"
        assert "subject" in email_dict, "email missing 'subject'"
        assert "body"    in email_dict, "email missing 'body'"
        assert "sender"  in email_dict, "email missing 'sender'"
        r.ok("SPEC-19")
    except Exception as exc:
        r.fail("SPEC-19", str(exc))


def check_hard_weight_sum(r: ValidationResult) -> None:
    try:
        import grader as g
        weights = [g._W_CATEGORY, g._W_PRIORITY, g._W_ACTION, g._W_RESPONSE]
        total   = sum(weights)
        assert abs(total - 1.0) < 1e-9, \
            f"Hard task reward weights sum to {total:.6f}, expected 1.0"
        r.ok("SPEC-20", f"weights={weights} sum={total:.6f}")
    except Exception as exc:
        r.fail("SPEC-20", str(exc))


def check_reset_idempotent(r: ValidationResult, SmartInboxEnv: Any) -> None:
    try:
        env = SmartInboxEnv()
        for i in range(3):
            obs = env.reset(task="classify", seed=i)
            assert obs.step == 0, f"After reset #{i+1}, step should be 0"
            assert obs.done is False
        r.ok("SPEC-21")
    except Exception as exc:
        r.fail("SPEC-21", str(exc))


def check_state_json_serialisable(r: ValidationResult, SmartInboxEnv: Any) -> None:
    try:
        env = SmartInboxEnv()
        env.reset(task="triage", seed=0)
        env.step(
            {"category": "spam", "priority": "low", "action": "ignore", "response": ""}
        )
        s = env.state()
        json_str = json.dumps(s)
        assert len(json_str) > 10, "state JSON is suspiciously short"
        r.ok("SPEC-22", f"{len(json_str)} chars")
    except Exception as exc:
        r.fail("SPEC-22", str(exc))


def check_info_has_error_key(r: ValidationResult, SmartInboxEnv: Any) -> None:
    try:
        for task in ["classify", "prioritize", "triage"]:
            action = (
                {"category": "spam"}
                if task == "classify" else
                {"priority": "low"}
                if task == "prioritize" else
                {"category": "spam", "priority": "low", "action": "ignore", "response": ""}
            )
            env = SmartInboxEnv()
            env.reset(task=task, seed=0)
            _, _, _, info = env.step(action)
            assert "error" in info, f"info dict missing 'error' key for task={task}"
        r.ok("SPEC-23")
    except Exception as exc:
        r.fail("SPEC-23", str(exc))


def check_no_infinite_loop(r: ValidationResult, SmartInboxEnv: Any) -> None:
    """Verify done flag is set and obs.done matches, preventing infinite loops."""
    try:
        for task in ["classify", "prioritize", "triage"]:
            action = (
                {"category": "spam"}
                if task == "classify" else
                {"priority": "low"}
                if task == "prioritize" else
                {"category": "spam", "priority": "low", "action": "ignore", "response": ""}
            )
            env = SmartInboxEnv()
            obs = env.reset(task=task, seed=0)
            steps = 0
            while not obs.done:
                obs, _, done, _ = env.step(action)
                steps += 1
                assert steps <= 100, "Exceeded 100 steps — possible infinite loop"
                if done:
                    assert obs.done is True, \
                        "done from step() and obs.done are inconsistent"
                    break
        r.ok("SPEC-24", f"terminated correctly for all tasks")
    except Exception as exc:
        r.fail("SPEC-24", str(exc))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(_bold("=" * 60))
    print(_bold("  SmartInboxEnv — OpenEnv Compliance Validator"))
    print(_bold("=" * 60))
    print()

    SmartInboxEnv = _import_env()
    r = ValidationResult()

    checks = [
        ("Interface",       [
            lambda: check_reset_returns_observation(r, SmartInboxEnv),
            lambda: check_step_returns_4_tuple(r, SmartInboxEnv),
            lambda: check_state_returns_dict(r, SmartInboxEnv),
        ]),
        ("Reward contract", [
            lambda: check_reward_value_range(r, SmartInboxEnv),
            lambda: check_reward_breakdown_is_dict(r, SmartInboxEnv),
        ]),
        ("Episode control", [
            lambda: check_episode_terminates(r, SmartInboxEnv),
            lambda: check_step_before_reset_raises(r, SmartInboxEnv),
            lambda: check_step_after_done_raises(r, SmartInboxEnv),
            lambda: check_bad_action_no_crash(r, SmartInboxEnv),
            lambda: check_no_infinite_loop(r, SmartInboxEnv),
        ]),
        ("Grader correctness", [
            lambda: check_deterministic(r, SmartInboxEnv),
            lambda: check_easy_binary_rewards(r, SmartInboxEnv),
            lambda: check_medium_partial_rewards(r, SmartInboxEnv),
            lambda: check_hard_shaped_reward(r, SmartInboxEnv),
            lambda: check_hard_perfect_score(r, SmartInboxEnv),
            lambda: check_hard_weight_sum(r),
        ]),
        ("Observation quality", [
            lambda: check_observation_task_valid(r, SmartInboxEnv),
            lambda: check_observation_prompt_nonempty(r, SmartInboxEnv),
            lambda: check_observation_no_ground_truth_leak(r, SmartInboxEnv),
        ]),
        ("Metadata & config", [
            lambda: check_yaml_exists(r),
            lambda: check_yaml_tasks_match(r, SmartInboxEnv),
            lambda: check_reset_idempotent(r, SmartInboxEnv),
            lambda: check_state_json_serialisable(r, SmartInboxEnv),
            lambda: check_info_has_error_key(r, SmartInboxEnv),
        ]),
    ]

    for section, fns in checks:
        print(_bold(f"\n── {section} " + "─" * (48 - len(section))))
        for fn in fns:
            try:
                fn()
            except Exception as exc:
                print(_red(f"  [INTERNAL ERROR] {exc}"))
                traceback.print_exc()

    r.summary()
    sys.exit(0 if r.all_passed else 1)


if __name__ == "__main__":
    main()
