# SmartInboxEnv v2

> **AI Email Triage & Decision System** — a fully OpenEnv-compliant benchmark for evaluating LLM agents on real-world enterprise email processing.

---

## Why this matters

Every company's inbox is a decision surface. Emails arrive carrying wildly different urgency, origin, and intent — a phishing attempt sits next to a CEO directive, a routine newsletter beside a data breach notification. Getting triage right has direct, measurable business impact:

- **False negatives on urgent mail** → missed SLA, lost revenue, legal exposure
- **False positives on spam** → wasted escalation cost, alert fatigue
- **Wrong action choices** → replying to phishing, ignoring legal notices, escalating newsletters

SmartInboxEnv captures this complexity as a rigorous, reproducible AI benchmark. It is designed to:

1. **Train** agents that generalise across email types and tones
2. **Evaluate** LLM reasoning on multi-step, structured decision-making
3. **Compare** models on a realistic, real-world task — not a toy classification problem

---

## Project structure

```
smart_inbox_env/
├── env.py                 # OpenEnv environment — step() / reset() / state()
├── models.py              # Typed models — pydantic v2 or stdlib dataclasses
├── tasks.py               # 30-email synthetic dataset + prompt templates
├── grader.py              # Deterministic graders for all three tasks
├── inference.py           # Baseline inference script (OpenAI-compatible API)
├── openenv_validator.py   # 24-check compliance validator (no dependencies)
├── openenv.yaml           # OpenEnv metadata specification
├── Dockerfile             # Python 3.10 image; validator runs at build time
├── requirements.txt       # pydantic>=2 + openai>=1.14
└── README.md
```

---

## Dataset

**30 diverse synthetic emails** covering:

| Type | Count | Category | Priority |
|---|---|---|---|
| Ops / infrastructure alerts | 4 | urgent | high |
| Security incidents & breaches | 3 | urgent | high |
| Legal notices & HR complaints | 2 | urgent | high |
| VIP / executive mail (CEO, VP) | 2 | urgent | high |
| Phishing & financial scams | 5 | spam | low |
| Angry complaints & refunds | 2 | normal | medium |
| Finance / compliance / audit | 3 | normal | medium |
| Routine business correspondence | 5 | normal | medium/low |
| Newsletters & FYI notices | 3 | normal | low |
| Ambiguous / mixed-intent | 1 | normal | medium |

All emails are fictional. Sampling is deterministic via `random.Random(seed)`.

---

## Tasks

### 🟢 Easy — `classify`

| | |
|---|---|
| **Input** | Email subject + body |
| **Output** | `"spam"` · `"urgent"` · `"normal"` |
| **Reward** | 1.0 correct / 0.0 incorrect |
| **Max steps** | 1 |

Binary classification. The agent must distinguish unsolicited/phishing mail from time-critical alerts and routine business email.

---

### 🟡 Medium — `prioritize`

| | |
|---|---|
| **Input** | Email subject + body |
| **Output** | `"low"` · `"medium"` · `"high"` |
| **Reward** | 1.0 exact / 0.5 off-by-one / 0.0 opposite extreme |
| **Max steps** | 1 |

Partial credit reflects that adjacent priority errors are far cheaper than catastrophic misclassification (treating a CEO email as low priority).

---

### 🔴 Hard — `triage`

| | |
|---|---|
| **Input** | Email subject + body |
| **Output** | classify + priority + action + response text |
| **Reward** | Shaped (see below) |
| **Max steps** | 1 |

The agent must reason across four dimensions simultaneously and produce a professional draft response.

---

## Reward design

### Hard task reward formula

```
base_score = (classification × 0.30)
           + (priority       × 0.20)
           + (action         × 0.20)
           + (response_quality × 0.30)
```

### Dependency penalties (applied after base)

| Condition | Penalty |
|---|---|
| Wrong classification | `× 0.60` multiplier — cascading error, the agent misunderstood the email type |
| Wrong action chosen | response score capped at 0.60 — correct wording can't rescue a wrong decision |
| VIP sender (CEO/VP/Director) not escalated | `− 0.10` deduction |

### Response quality scoring

Response quality uses keyword coverage — deterministic, no LLM judge required:

- **IGNORE emails** (spam/newsletters): full marks for empty string; penalised for any reply
- **Keywords defined**: fraction of ideal keywords present in response
- **No keywords**: binary — substantive response (>20 chars) = 1.0

### Why shaped rewards?

Binary rewards (pass/fail) don't train good agents. Shaped rewards:

1. Signal *which dimension* failed, not just *that* it failed
2. Give partial credit for partially-correct reasoning (e.g. right action, wrong priority)
3. Penalise cascading errors more severely (wrong classification → wrong everything)
4. Create meaningful gradient signal for RL training

---

## Intelligence rules

The environment encodes real-world business logic that agents must learn:

```
VIP sender (ceo@, vp.*, director@, board@)
  → always HIGH priority
  → always ESCALATE action
  → penalty if agent fails this

Phishing signals (suspicious domain + urgency + link)
  → always SPAM category
  → always IGNORE action

Legal / security / breach
  → always URGENT category
  → always ESCALATE action

Spam / newsletter
  → LOW priority
  → IGNORE action
  → empty response required
```

---

## Setup

**Requirements:** Python 3.9+, pip, OpenAI-compatible API endpoint.

```bash
pip install -r requirements.txt
```

**Verify OpenEnv compliance (no API needed):**

```bash
python openenv_validator.py
# Expected: 38/38 PASSED — OpenEnv compliant!
```

---

## Running inference

### Set environment variables

```bash
# OpenAI
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...

# HuggingFace (HF_TOKEN takes precedence)
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3-8b-instruct
export HF_TOKEN=hf_...

# Local / Ollama
export API_BASE_URL=http://localhost:11434/v1
export MODEL_NAME=llama3
export OPENAI_API_KEY=ollama
```

### Run all tasks

```bash
python inference.py --task all --seed 42
```

### Run one task

```bash
python inference.py --task classify    --seed 42
python inference.py --task prioritize  --seed 42
python inference.py --task triage      --seed 42
```

### Expected output

```
[START] task=classify env=SmartInboxEnv model=gpt-4o-mini
[STEP] step=1 action={"category":"urgent"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00

[START] task=prioritize env=SmartInboxEnv model=gpt-4o-mini
[STEP] step=1 action={"priority":"high"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00

[START] task=triage env=SmartInboxEnv model=gpt-4o-mini
[STEP] step=1 action={"category":"urgent","priority":"high","action":"escalate","response":"This has been escalated to the on-call engineering manager immediately."} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00
```

---

## Using the environment directly

```python
from env import SmartInboxEnv

env = SmartInboxEnv()

# Easy task
obs = env.reset("classify", seed=42)
obs, reward, done, info = env.step({"category": "urgent"})
print(reward.value)       # 1.0 or 0.0
print(reward.feedback)

# Medium task
obs = env.reset("prioritize", seed=42)
obs, reward, done, info = env.step({"priority": "high"})
print(reward.value)       # 0.0, 0.5, or 1.0

# Hard task
obs = env.reset("triage", seed=42)
print(obs.prompt)
obs, reward, done, info = env.step({
    "category": "urgent",
    "priority": "high",
    "action":   "escalate",
    "response": "Escalating to the engineering manager immediately.",
})
print(reward.value)          # 0.0 – 1.0
print(reward.breakdown)      # per-dimension scores
print(reward.feedback)       # human-readable explanation
print(env.state())           # full episode state
```

---

## Docker

### Build (runs validator — build fails if non-compliant)

```bash
docker build -t smart-inbox-env .
```

### Run all tasks

```bash
docker run --rm \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e OPENAI_API_KEY=sk-... \
  smart-inbox-env
```

### Override task or seed

```bash
docker run --rm \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e OPENAI_API_KEY=sk-... \
  smart-inbox-env python inference.py --task triage --seed 7
```

### Run validator only

```bash
docker run --rm --entrypoint python smart-inbox-env openenv_validator.py
```

---

## Reproducibility

All email sampling uses `random.Random(seed)`. The same `--seed` value always selects the same email. The task name is XOR'd into the seed so `classify seed=42` and `triage seed=42` select different emails. This makes cross-model comparisons fully fair.

---

## Extending the environment

| Goal | Where |
|---|---|
| Add emails | Append `Email(...)` to `EMAILS` in `tasks.py` |
| Adjust prompts | Edit `TASK_PROMPTS` in `tasks.py` |
| Change reward weights | Edit `_W_*` constants at top of `grader.py` |
| Add penalty rules | Extend `grade_hard()` in `grader.py` |
| Add new VIP patterns | Extend `VIP_SENDER_PREFIXES` in `tasks.py` |
| Add new task | Add to `TaskName`, `TASK_PROMPTS`, `MAX_STEPS`; add grader; update `env.py._parse_action` |

---

## OpenEnv compliance

Passes all 38 automated checks across 24 specifications:

```
SPEC-01  reset() returns Observation with all required fields
SPEC-02  step() returns (Observation, Reward, bool, dict)
SPEC-03  state() returns dict with required keys
SPEC-04  Reward.value is float in [0.0, 1.0]
SPEC-05  Reward.breakdown is a dict
SPEC-06  Episode terminates after max_steps
SPEC-07  step() raises RuntimeError before reset()
SPEC-08  step() raises RuntimeError after episode ends
SPEC-09  Bad action → 0.0 reward + done=True (no crash)
SPEC-10  Deterministic: same seed → same score ×3 runs
SPEC-11  Easy task: binary rewards only (0.0 or 1.0)
SPEC-12  Medium task: scores in {0.0, 0.5, 1.0}
SPEC-13  Hard task: shaped reward (multiple distinct values)
SPEC-14  Hard task: perfect action scores ≥ 0.9 on all 30 emails
SPEC-15  openenv.yaml exists with required keys
SPEC-16  YAML tasks match implemented tasks
SPEC-17  Observation.task is a valid TaskName
SPEC-18  Observation.prompt is non-empty string
SPEC-19  No ground-truth leaked in observation
SPEC-20  Hard task reward weights sum to 1.0
SPEC-21  reset() is idempotent
SPEC-22  state() is JSON-serialisable
SPEC-23  info dict from step() contains 'error' key
SPEC-24  No infinite loop: done flag consistent
```

---

## License

MIT
