#!/usr/bin/env python3
"""
server.py
---------
OpenEnv-compliant HTTP REST server for SmartInboxEnv.

The hackathon platform (Scaler OpenEnv) validates environments by making
HTTP calls to this server. All required endpoints are implemented here.

Endpoints (OpenEnv spec)
------------------------
POST   /reset       body: {"task": str, "seed": int|null}
                    → Observation JSON

POST   /step        body: action dict  e.g. {"category": "spam"}
                    → {"observation": ..., "reward": ..., "done": bool, "info": ...}

GET    /state       → full environment state dict

GET    /health      → {"status": "ok", "env": "SmartInboxEnv", "version": "2.0.0"}

GET    /            → same as /health  (root ping)

GET    /info        → environment metadata from openenv.yaml

Run
---
python server.py                 # default port 8000
PORT=5000 python server.py       # custom port
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional, Tuple

# ── Use Flask (zero extra install needed in the Docker image) ─────────────────
try:
    from flask import Flask, Response, jsonify, request
except ImportError:
    print("[ERROR] Flask not found. Install with: pip install flask>=2.3", file=sys.stderr)
    sys.exit(1)

from env import SmartInboxEnv
from models import TaskName

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# One shared environment instance per server process.
# The platform resets it between episodes via POST /reset.
_env = SmartInboxEnv()

ENV_NAME    = "SmartInboxEnv"
ENV_VERSION = "2.0.0"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    """Convert Observation (pydantic or dataclass) to a plain dict."""
    if hasattr(obs, "model_dump"):          # pydantic v2
        d = obs.model_dump()
        d["task"] = d["task"].value if hasattr(d["task"], "value") else d["task"]
        return d
    if hasattr(obs, "to_dict"):
        return obs.to_dict()
    # dataclass fallback
    return {
        "task":   obs.task.value if hasattr(obs.task, "value") else obs.task,
        "step":   obs.step,
        "email":  obs.email,
        "prompt": obs.prompt,
        "done":   obs.done,
        "info":   obs.info,
    }


def _reward_to_dict(reward: Any) -> Dict[str, Any]:
    """Convert Reward to a plain dict."""
    if hasattr(reward, "model_dump"):
        return reward.model_dump()
    if hasattr(reward, "to_dict"):
        return reward.to_dict()
    return {
        "value":     reward.value,
        "breakdown": reward.breakdown,
        "feedback":  reward.feedback,
    }


def _success_response(data: Dict[str, Any], status: int = 200) -> Response:
    return Response(
        json.dumps(data, ensure_ascii=False),
        status=status,
        mimetype="application/json",
    )


def _error_response(message: str, status: int = 400) -> Response:
    return Response(
        json.dumps({"error": message}),
        status=status,
        mimetype="application/json",
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
def health() -> Response:
    """Platform liveness / readiness probe."""
    return _success_response({
        "status":  "ok",
        "env":     ENV_NAME,
        "version": ENV_VERSION,
        "tasks":   [t.value for t in TaskName],
    })


@app.get("/info")
def info() -> Response:
    """Return environment metadata."""
    return _success_response({
        "name":        ENV_NAME,
        "version":     ENV_VERSION,
        "description": "AI email triage and decision-making benchmark environment.",
        "tasks": [
            {
                "name":        "classify",
                "difficulty":  "easy",
                "reward_type": "deterministic",
                "max_steps":   1,
            },
            {
                "name":        "prioritize",
                "difficulty":  "medium",
                "reward_type": "partial",
                "max_steps":   1,
            },
            {
                "name":        "triage",
                "difficulty":  "hard",
                "reward_type": "shaped",
                "max_steps":   1,
            },
        ],
    })


@app.post("/reset")
def reset() -> Response:
    """
    Reset the environment and return the initial observation.

    Body (JSON, all fields optional):
      {
        "task": "classify" | "prioritize" | "triage",   default: "classify"
        "seed": <integer>                                default: null
      }
    """
    try:
        body: Dict[str, Any] = request.get_json(silent=True) or {}
        task = str(body.get("task", "classify")).strip()
        seed = body.get("seed", None)
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None

        obs = _env.reset(task=task, seed=seed)
        return _success_response(_obs_to_dict(obs))

    except ValueError as exc:
        return _error_response(str(exc), status=400)
    except Exception as exc:
        traceback.print_exc()
        return _error_response(f"Internal error during reset: {exc}", status=500)


@app.post("/step")
def step() -> Response:
    """
    Execute one environment step.

    Body (JSON): action dict for the current task.
      classify   → {"category": "spam|urgent|normal"}
      prioritize → {"priority": "low|medium|high"}
      triage     → {"category": ..., "priority": ..., "action": ..., "response": ...}

    Returns:
      {
        "observation": {...},
        "reward":      {"value": float, "breakdown": {...}, "feedback": str},
        "done":        bool,
        "info":        {...}
      }
    """
    try:
        action_dict: Dict[str, Any] = request.get_json(silent=True) or {}

        obs, reward, done, info = _env.step(action_dict)

        return _success_response({
            "observation": _obs_to_dict(obs),
            "reward":      _reward_to_dict(reward),
            "done":        done,
            "info":        info,
        })

    except RuntimeError as exc:
        # step() before reset(), or step() after done
        return _error_response(str(exc), status=400)
    except Exception as exc:
        traceback.print_exc()
        return _error_response(f"Internal error during step: {exc}", status=500)


@app.get("/state")
def state() -> Response:
    """Return the full current environment state."""
    try:
        return _success_response(_env.state())
    except Exception as exc:
        traceback.print_exc()
        return _error_response(f"Internal error: {exc}", status=500)


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e: Any) -> Response:
    return _error_response(
        f"Endpoint not found. Available: GET /, GET /health, GET /info, "
        f"GET /state, POST /reset, POST /step",
        status=404,
    )


@app.errorhandler(405)
def method_not_allowed(e: Any) -> Response:
    return _error_response(
        "Method not allowed. Check POST vs GET for this endpoint.",
        status=405,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    print(f"[SmartInboxEnv] Starting HTTP server on {host}:{port}", flush=True)
    print(f"[SmartInboxEnv] Endpoints: POST /reset  POST /step  GET /state  GET /health", flush=True)

    app.run(host=host, port=port, debug=debug)
