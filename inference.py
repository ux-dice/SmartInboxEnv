#!/usr/bin/env python3
"""
inference.py
------------
Baseline inference script for SmartInboxEnv.

Environment variables:
  API_BASE_URL   — OpenAI-compatible base URL  (required)
  MODEL_NAME     — model identifier            (required)
  HF_TOKEN       — HuggingFace token           (optional; takes precedence)
  OPENAI_API_KEY — OpenAI API key              (optional fallback)

Exact stdout format (OpenEnv spec):
  [START] task=<task_name> env=SmartInboxEnv model=<model_name>
  [STEP] step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>

Usage:
  python inference.py [--task classify|prioritize|triage|all] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print(
        "[ERROR] 'openai' package not found. Install with: pip install openai>=1.14.0",
        file=sys.stderr,
    )
    sys.exit(1)

from env import SmartInboxEnv
from tasks import ALL_TASKS

# ── Constants ──────────────────────────────────────────────────────────────────

ENV_NAME     = "SmartInboxEnv"
DEFAULT_SEED = 42

SYSTEM_PROMPT = (
    "You are a professional email triage assistant. "
    "You MUST respond with valid JSON only — no markdown fences, no backticks, "
    "no explanations, no preamble. Output only the raw JSON object."
)


# ── Env var helpers ────────────────────────────────────────────────────────────

def require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(f"[ERROR] Required environment variable '{name}' is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def optional_env(name: str) -> Optional[str]:
    val = os.environ.get(name, "").strip()
    return val if val else None


# ── JSON cleaning ──────────────────────────────────────────────────────────────

def clean_json(raw: str) -> str:
    """Strip markdown code fences if the model wrapped the JSON."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop opening fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_model_output(raw: str, task_name: str) -> Dict[str, Any]:
    """
    Parse and normalise model output into an action dict.
    Raises ValueError with a clear message on failure.
    """
    cleaned = clean_json(raw)
    try:
        action = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model returned invalid JSON: {exc}. "
            f"Raw output (first 200 chars): {raw[:200]!r}"
        )

    if not isinstance(action, dict):
        raise ValueError(f"Expected JSON object, got {type(action).__name__}. Raw: {raw[:200]!r}")

    # Normalise enum fields to lowercase
    for key in ("category", "priority", "action"):
        if key in action and isinstance(action[key], str):
            action[key] = action[key].lower().strip()

    # Ensure response field exists for triage
    if task_name == "triage" and "response" not in action:
        action["response"] = ""

    return action


# ── Model call ────────────────────────────────────────────────────────────────

def call_model(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


# ── Log formatting ─────────────────────────────────────────────────────────────

def fmt_action(d: Dict[str, Any]) -> str:
    """Compact JSON, no spaces."""
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def fmt_rewards(rewards: List[float]) -> str:
    """Comma-separated, exactly 2 decimal places."""
    return ",".join(f"{r:.2f}" for r in rewards)


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(
    step_n: int,
    action_dict: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_str = error if error is not None else "null"
    if error_str != "null" and len(error_str) > 200:
        error_str = error_str[:197] + "..."
    print(
        f"[STEP] step={step_n} "
        f"action={fmt_action(action_dict)} "
        f"reward={reward:.2f} "
        f"done={'true' if done else 'false'} "
        f"error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps} "
        f"score={score:.2f} "
        f"rewards={fmt_rewards(rewards)}",
        flush=True,
    )


# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(
    task_name: str,
    client: OpenAI,
    model_name: str,
    seed: int,
) -> None:
    """Run one episode; all exceptions are caught and surfaced in [STEP] error field."""
    env = SmartInboxEnv()
    obs = env.reset(task=task_name, seed=seed)
    rewards: List[float] = []
    steps_taken = 0

    log_start(task_name, model_name)

    while not obs.done:
        step_n      = obs.step + 1
        error_msg: Optional[str]  = None
        reward_val  = 0.0
        done        = False
        action_dict: Dict[str, Any] = {}

        # Call model
        try:
            raw         = call_model(client, model_name, obs.prompt)
            action_dict = parse_model_output(raw, task_name)
        except Exception as exc:
            error_msg   = str(exc).replace("\n", " ")
            done        = True

        # Step environment (only on successful parse)
        if error_msg is None:
            try:
                obs, reward_obj, done, info = env.step(action_dict)
                reward_val = reward_obj.value
                if info.get("error"):
                    error_msg = str(info["error"])
            except Exception as exc:
                error_msg = str(exc).replace("\n", " ")
                done      = True

        rewards.append(reward_val)
        steps_taken += 1
        log_step(step_n, action_dict, reward_val, done, error_msg)

        if done:
            break

    final_score = rewards[-1] if rewards else 0.0
    success     = bool(rewards) and final_score >= 0.5
    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


def run_all_tasks(client: OpenAI, model_name: str, seed: int) -> None:
    """Run all three tasks in order; blank line between each."""
    for task in ALL_TASKS:
        run_task(task, client, model_name, seed)
        print("", flush=True)


# ── Client factory ────────────────────────────────────────────────────────────

def build_client() -> Tuple[OpenAI, str]:
    api_base   = require_env("API_BASE_URL")
    model_name = require_env("MODEL_NAME")
    hf_token   = optional_env("HF_TOKEN")
    oai_key    = optional_env("OPENAI_API_KEY")
    api_key    = hf_token or oai_key or "none"
    return OpenAI(base_url=api_base, api_key=api_key), model_name


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SmartInboxEnv — OpenEnv baseline inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Required env vars: API_BASE_URL, MODEL_NAME\n"
            "Optional env vars: HF_TOKEN, OPENAI_API_KEY"
        ),
    )
    parser.add_argument(
        "--task",
        choices=["classify", "prioritize", "triage", "all"],
        default="all",
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    args   = parser.parse_args()
    client, model_name = build_client()

    if args.task == "all":
        run_all_tasks(client, model_name, args.seed)
    else:
        run_task(args.task, client, model_name, args.seed)


if __name__ == "__main__":
    main()
