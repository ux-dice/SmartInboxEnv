"""
tasks.py
--------
Synthetic email dataset (30 emails) and task configuration for SmartInboxEnv.

Covers: ops alerts, security incidents, phishing, CEO/VIP mail, angry
complaints, refund requests, legal/HR issues, ambiguous edge-cases,
technical failures, newsletters, and routine business correspondence.

VIP senders  (ceo@, boss@, board@, vp., director@, manager@) are always HIGH
priority and require escalation.
Phishing     (suspicious domains, urgency+link patterns) are always SPAM.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from models import ActionType, Email, EmailCategory, Priority, TaskName


# ── Synthetic email dataset (30 emails) ───────────────────────────────────────

EMAILS: List[Email] = [

    # ── URGENT / HIGH / ESCALATE ──────────────────────────────────────────────

    Email(
        id="e001",
        subject="URGENT: Production API server returning 503",
        body=(
            "Hi team,\n\n"
            "Our main API server has been returning 503 errors for the last 8 minutes. "
            "Customers cannot complete checkout and we are losing revenue at ~$3,000/min. "
            "The on-call engineer has not responded to PagerDuty. "
            "Please escalate immediately to the engineering manager."
        ),
        sender="ops-alert@company.com",
        true_category=EmailCategory.URGENT,
        true_priority=Priority.HIGH,
        true_action=ActionType.ESCALATE,
        ideal_response_keywords=["escalate", "engineering", "investigating", "immediately", "on-call"],
    ),

    Email(
        id="e002",
        subject="Security alert: unauthorised admin login detected",
        body=(
            "ALERT: An unauthorised login was detected on admin account 'j.smith' "
            "from IP 185.234.101.9 (Russia) at 02:14 UTC, preceded by 47 failed attempts. "
            "The account has NOT been locked. Initiate incident response and lock the "
            "account immediately to prevent data exfiltration."
        ),
        sender="security@company.com",
        true_category=EmailCategory.URGENT,
        true_priority=Priority.HIGH,
        true_action=ActionType.ESCALATE,
        ideal_response_keywords=["lock", "incident", "security", "escalate", "immediately", "protocol"],
    ),

    Email(
        id="e003",
        subject="Client escalation — Apex Corp threatening contract termination",
        body=(
            "I'm writing on behalf of Apex Corp (£2.4M ARR) who has raised a formal "
            "complaint about delayed deliverables on Project Phoenix. The project is "
            "3 weeks behind schedule. They will terminate the contract if senior leadership "
            "does not contact them by 09:00 tomorrow. This is extremely time-sensitive."
        ),
        sender="account.manager@company.com",
        true_category=EmailCategory.URGENT,
        true_priority=Priority.HIGH,
        true_action=ActionType.ESCALATE,
        ideal_response_keywords=["escalate", "leadership", "contact", "priority", "immediately", "client"],
    ),

    Email(
        id="e004",
        subject="CRITICAL: Data centre fire suppression activated — Rack Room B",
        body=(
            "CRITICAL ALERT: The automated fire suppression system triggered in Rack Room B "
            "at 14:32 UTC. All servers in rows B4–B12 have been powered down. "
            "ALL engineering staff must evacuate Rack Room B immediately. "
            "Do not re-enter until the all-clear is given. This is NOT a drill."
        ),
        sender="facilities-alert@company.com",
        true_category=EmailCategory.URGENT,
        true_priority=Priority.HIGH,
        true_action=ActionType.ESCALATE,
        ideal_response_keywords=["evacuate", "facilities", "escalate", "critical", "immediately", "safety"],
    ),

    Email(
        id="e005",
        subject="Legal notice — cease and desist from Thornton & Associates",
        body=(
            "Dear Sir/Madam,\n\n"
            "We represent GlobalStream Media and hereby issue a formal cease and desist "
            "regarding your unauthorised use of copyrighted content. You have 72 hours to "
            "comply and remove the infringing material, or we will seek injunctive relief "
            "and file for damages. Please forward this to your legal counsel immediately."
        ),
        sender="legal@thornton-associates.com",
        true_category=EmailCategory.URGENT,
        true_priority=Priority.HIGH,
        true_action=ActionType.ESCALATE,
        ideal_response_keywords=["legal", "counsel", "escalate", "acknowledge", "72 hours"],
    ),

    Email(
        id="e006",
        subject="HR complaint — hostile workplace allegation filed",
        body=(
            "I am formally raising a complaint about a hostile work environment in the "
            "Engineering department. The behaviour of my direct manager has made it "
            "impossible to work. I am prepared to escalate to the Employment Tribunal "
            "if this is not addressed within 5 business days. Please acknowledge receipt."
        ),
        sender="employee.complaint@company-staff.com",
        true_category=EmailCategory.URGENT,
        true_priority=Priority.HIGH,
        true_action=ActionType.ESCALATE,
        ideal_response_keywords=["acknowledge", "HR", "escalate", "investigating", "formally"],
    ),

    # ── VIP senders — always HIGH priority, always ESCALATE ──────────────────

    Email(
        id="e007",
        subject="Strategy review meeting — need your input this week",
        body=(
            "Hi,\n\n"
            "I need a full briefing on the Q4 product roadmap before our board presentation "
            "on Friday. Please compile the slides and metrics and send them to me directly "
            "by Thursday EOD. This is a board-level deliverable."
        ),
        sender="ceo@company.com",
        true_category=EmailCategory.URGENT,
        true_priority=Priority.HIGH,
        true_action=ActionType.ESCALATE,
        ideal_response_keywords=["board", "Friday", "CEO", "priority", "confirm", "prepare"],
    ),

    Email(
        id="e008",
        subject="Vendor contract renewal — decision required",
        body=(
            "Team,\n\n"
            "The SaaS vendor contract expires in 10 days. We need a final decision on "
            "renewal vs migration by tomorrow morning. I've reviewed the proposals — "
            "please send me your recommendation today so I can sign off before EOD."
        ),
        sender="vp.operations@company.com",
        true_category=EmailCategory.URGENT,
        true_priority=Priority.HIGH,
        true_action=ActionType.ESCALATE,
        ideal_response_keywords=["decision", "confirm", "renewal", "escalate", "today"],
    ),

    # ── SPAM / PHISHING ───────────────────────────────────────────────────────

    Email(
        id="e009",
        subject="You've won a FREE iPhone 15 Pro! Claim now",
        body=(
            "Congratulations! You have been randomly selected to receive a FREE iPhone 15 Pro. "
            "Click the link below to claim your prize. This offer expires in 24 hours. "
            "No purchase necessary. Act now — only 3 prizes remaining!"
        ),
        sender="prizes@totally-legit-giveaway.biz",
        true_category=EmailCategory.SPAM,
        true_priority=Priority.LOW,
        true_action=ActionType.IGNORE,
        ideal_response_keywords=[],
    ),

    Email(
        id="e010",
        subject="TRIPLE your productivity with our revolutionary supplement!!!",
        body=(
            "Dear Valued Professional, Are you tired of low energy and poor focus? "
            "Our clinically-PROVEN NeuroBoost supplement will TRIPLE your output in just 7 days! "
            "Order NOW for 90% OFF. ONLY 12 bottles left. Don't miss this ONCE-IN-A-LIFETIME deal!!!"
        ),
        sender="deals@spamcentral.net",
        true_category=EmailCategory.SPAM,
        true_priority=Priority.LOW,
        true_action=ActionType.IGNORE,
        ideal_response_keywords=[],
    ),

    Email(
        id="e011",
        subject="Your account has been SUSPENDED — verify now to restore access",
        body=(
            "Dear Customer, your account has been suspended due to unusual activity. "
            "Click here immediately to verify your identity and restore access: "
            "http://secure-account-verify.xyz/login?token=abc123. "
            "Failure to verify within 24 hours will result in permanent account closure."
        ),
        sender="security-noreply@paypa1-verify.xyz",
        true_category=EmailCategory.SPAM,
        true_priority=Priority.LOW,
        true_action=ActionType.IGNORE,
        ideal_response_keywords=[],
    ),

    Email(
        id="e012",
        subject="Exclusive investment opportunity — guaranteed 40% returns",
        body=(
            "Dear Investor, I am reaching out with an exclusive opportunity to join our "
            "private fund that has delivered 40% annual returns with ZERO risk. "
            "This offer is available to only 50 investors. Wire $10,000 today to secure your spot. "
            "Reply with your bank details to get started."
        ),
        sender="invest@offshore-wealth-mgmt.bz",
        true_category=EmailCategory.SPAM,
        true_priority=Priority.LOW,
        true_action=ActionType.IGNORE,
        ideal_response_keywords=[],
    ),

    Email(
        id="e013",
        subject="CEO WIRE TRANSFER REQUEST — confidential",
        body=(
            "This is a confidential request from the CEO. We are closing an acquisition deal "
            "today and require an urgent wire transfer of $85,000 to the following account. "
            "Do NOT discuss this with anyone — it is highly sensitive. "
            "Please process immediately: IBAN DE89 3704 0044 0532 0130 00."
        ),
        sender="ceo.office@company-corp.net",
        true_category=EmailCategory.SPAM,
        true_priority=Priority.LOW,
        true_action=ActionType.IGNORE,
        ideal_response_keywords=[],
    ),

    # ── NORMAL / MEDIUM / REPLY ───────────────────────────────────────────────

    Email(
        id="e014",
        subject="Q3 financial report — review required by Friday",
        body=(
            "Hi,\n\n"
            "A reminder that the Q3 financial report requires your sign-off by end of day "
            "Friday. Please review pages 4–7 (revenue reconciliation) and add your approval "
            "in the shared Google Doc. The board presentation is Monday morning. Thank you."
        ),
        sender="finance@company.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.MEDIUM,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["review", "Friday", "confirm", "sign-off", "acknowledge"],
    ),

    Email(
        id="e015",
        subject="Invoice #4892 — 30 days overdue, service suspension imminent",
        body=(
            "Dear Accounts Payable,\n\n"
            "Invoice #4892 for £12,450 (due 2024-10-01) remains unpaid — 30 days overdue. "
            "We will suspend your service access and apply a 5% late fee if payment is "
            "not received within 48 hours. Please process the payment or contact us "
            "to discuss a payment plan."
        ),
        sender="billing@vendor-partner.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.MEDIUM,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["payment", "process", "invoice", "48 hours", "accounts"],
    ),

    Email(
        id="e016",
        subject="Angry customer complaint — unacceptable service experience",
        body=(
            "I am absolutely furious. I have been a customer for 5 years and your support "
            "team has ignored my last THREE emails about a billing error that overcharged "
            "me £340. I want this resolved TODAY or I will be posting a detailed review "
            "on Trustpilot and contacting my bank for a chargeback."
        ),
        sender="james.miller@gmail.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.MEDIUM,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["apologise", "resolve", "billing", "investigate", "contact", "urgently"],
    ),

    Email(
        id="e017",
        subject="Refund request — order #ORD-88921 not delivered",
        body=(
            "Hi,\n\n"
            "I placed order #ORD-88921 on 10th October and it still hasn't arrived. "
            "The tracking shows it's been 'in transit' for 18 days. "
            "I'd like a full refund processed back to my original payment method please. "
            "Can you confirm receipt of this request?"
        ),
        sender="customer@example.co.uk",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.MEDIUM,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["refund", "order", "investigate", "confirm", "process", "apologise"],
    ),

    Email(
        id="e018",
        subject="New employee onboarding — system access required",
        body=(
            "Hi IT,\n\n"
            "We have a new Software Engineer, Ana Torres, starting on Monday 4th November. "
            "She'll need a laptop provisioned with the standard dev environment, "
            "access to GitHub, Jira, Slack, and the VPN. "
            "Please confirm you've received this request. Thanks."
        ),
        sender="hr@company.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.MEDIUM,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["confirm", "access", "provision", "Monday", "onboarding"],
    ),

    # ── NORMAL / LOW / REPLY ──────────────────────────────────────────────────

    Email(
        id="e019",
        subject="Team lunch — Tuesday 12:30pm, The Corner Bistro",
        body=(
            "Hey everyone,\n\n"
            "We're booking a team lunch for next Tuesday at 12:30pm at The Corner Bistro. "
            "Please reply by Monday so we can give the restaurant an accurate headcount. "
            "Looking forward to seeing everyone!"
        ),
        sender="sarah.jones@company.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.LOW,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["confirm", "attend", "Tuesday", "headcount"],
    ),

    Email(
        id="e020",
        subject="Office plants watering schedule — volunteers needed",
        body=(
            "Hi all,\n\n"
            "As you may have noticed, the office plants are looking a bit sorry for themselves! "
            "We're setting up a rotating watering schedule and are looking for volunteers. "
            "Please reply if you'd be happy to take a weekly slot. "
            "Sign-up sheet is also on the noticeboard. Thanks!"
        ),
        sender="facilities@company.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.LOW,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["volunteer", "confirm", "schedule"],
    ),

    # ── NORMAL / LOW / IGNORE ─────────────────────────────────────────────────

    Email(
        id="e021",
        subject="October company newsletter",
        body=(
            "Welcome to the October edition of the company newsletter! "
            "This month: office renovation updates, Employee of the Month Priya Sharma, "
            "charity 10K highlights, and the upcoming all-hands preview. "
            "Enjoy the read and have a great week!"
        ),
        sender="comms@company.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.LOW,
        true_action=ActionType.IGNORE,
        ideal_response_keywords=[],
    ),

    Email(
        id="e022",
        subject="Reminder: submit your timesheet by Friday 5pm",
        body=(
            "Hi everyone,\n\n"
            "Just a friendly reminder to submit your timesheets in the HR portal "
            "by 5pm this Friday. If you have any issues accessing the portal, "
            "please contact hr-support@company.com. Thank you!"
        ),
        sender="payroll@company.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.LOW,
        true_action=ActionType.IGNORE,
        ideal_response_keywords=[],
    ),

    # ── AMBIGUOUS / MIXED-INTENT edge cases ───────────────────────────────────

    Email(
        id="e023",
        subject="System performance degradation — some users affected",
        body=(
            "Hi,\n\n"
            "We're seeing some intermittent slowness in the reporting module — "
            "around 15% of users in the EU region are experiencing 5–10 second delays. "
            "The team is investigating. Not a full outage yet. "
            "Will update at next scheduled maintenance window unless it worsens."
        ),
        sender="monitoring@company.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.MEDIUM,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["acknowledged", "monitoring", "update", "investigating"],
    ),

    Email(
        id="e024",
        subject="Partnership proposal — AI integration opportunity",
        body=(
            "Dear Team,\n\n"
            "We at NovaTech AI would like to propose a strategic partnership to integrate "
            "our NLP platform with your product. We believe this could drive 20% revenue "
            "growth. I'd love to schedule a 30-minute call at your convenience. "
            "Please let me know your availability this week or next."
        ),
        sender="partnerships@novatech-ai.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.LOW,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["interest", "schedule", "availability", "discuss"],
    ),

    Email(
        id="e025",
        subject="Possible data breach — third-party vendor notification",
        body=(
            "Dear Customer,\n\n"
            "We are writing to inform you that our third-party payment processor "
            "experienced a security incident that may have exposed transaction data "
            "between 2024-08-01 and 2024-09-15. We are conducting a full investigation. "
            "We recommend you monitor your accounts and consider resetting your credentials."
        ),
        sender="security-notification@trustedpayments.com",
        true_category=EmailCategory.URGENT,
        true_priority=Priority.HIGH,
        true_action=ActionType.ESCALATE,
        ideal_response_keywords=["security", "breach", "escalate", "investigate", "credentials", "monitor"],
    ),

    Email(
        id="e026",
        subject="Regulatory compliance audit — documents required by next Wednesday",
        body=(
            "Dear Compliance Team,\n\n"
            "We are conducting our annual SOC 2 Type II audit and require the following "
            "documentation by Wednesday 8th November: access control policies, incident "
            "response logs (last 12 months), and vendor risk assessments. "
            "Failure to provide these by the deadline may result in a qualified opinion."
        ),
        sender="audit@external-auditors.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.MEDIUM,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["confirm", "documents", "compliance", "audit", "Wednesday"],
    ),

    Email(
        id="e027",
        subject="Request for proposal — new CRM platform evaluation",
        body=(
            "Hi,\n\n"
            "As part of our annual technology review, we are evaluating CRM platforms "
            "to replace our current Salesforce subscription. Could you please send us "
            "your pricing and feature comparison by end of month? "
            "We are looking at HubSpot, Pipedrive, and your solution."
        ),
        sender="procurement@prospect-corp.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.MEDIUM,
        true_action=ActionType.REPLY,
        ideal_response_keywords=["proposal", "pricing", "features", "send", "confirm"],
    ),

    Email(
        id="e028",
        subject="Database migration — scheduled maintenance this Saturday 02:00–06:00 UTC",
        body=(
            "Hi all,\n\n"
            "We will be migrating the primary PostgreSQL database to the new cluster "
            "this Saturday between 02:00–06:00 UTC. The application will be in read-only "
            "mode during this window. No action required from you. "
            "Please plan your work accordingly and contact db-team@ with any concerns."
        ),
        sender="db-team@company.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.LOW,
        true_action=ActionType.IGNORE,
        ideal_response_keywords=[],
    ),

    Email(
        id="e029",
        subject="Password reset confirmation",
        body=(
            "Hi,\n\n"
            "Your password was successfully reset on 2024-10-25 at 14:32 UTC. "
            "If you did not request this change, please contact security@company.com "
            "immediately. No further action is required if this was you."
        ),
        sender="noreply@company.com",
        true_category=EmailCategory.NORMAL,
        true_priority=Priority.LOW,
        true_action=ActionType.IGNORE,
        ideal_response_keywords=[],
    ),

    Email(
        id="e030",
        subject="Urgent: your account will be deleted unless you respond",
        body=(
            "URGENT! Your account is scheduled for deletion in 24 hours due to inactivity. "
            "To prevent this, click here NOW: http://account-save.ru/prevent-delete?id=99182. "
            "This is your FINAL NOTICE. Do not ignore this message."
        ),
        sender="account-alert@notifications-service.ru",
        true_category=EmailCategory.SPAM,
        true_priority=Priority.LOW,
        true_action=ActionType.IGNORE,
        ideal_response_keywords=[],
    ),
]

# Quick lookup by ID
EMAIL_BY_ID: Dict[str, Email] = {e.id: e for e in EMAILS}

# ── VIP sender detection ───────────────────────────────────────────────────────
# Emails from these prefixes are always HIGH priority and require escalation.
VIP_SENDER_PREFIXES = ("ceo@", "cto@", "cfo@", "coo@", "boss@", "board@",
                       "vp.", "vp-", "director@", "exec@")


def is_vip_sender(sender: str) -> bool:
    """Return True if the sender address indicates a VIP / executive."""
    local = sender.lower().split("@")[0]
    domain_part = sender.lower()
    return any(domain_part.startswith(p) or local == p.rstrip("@.-") or
               local.startswith(p.rstrip("@")) for p in VIP_SENDER_PREFIXES)


# ── Prompt templates ───────────────────────────────────────────────────────────

TASK_PROMPTS: Dict[str, str] = {
    TaskName.EASY.value: (
        "You are an expert email classifier. Read the email below and classify it.\n\n"
        "CATEGORIES — choose exactly one:\n"
        "  spam   — unsolicited, promotional, or phishing email\n"
        "  urgent — requires immediate attention or same-day action\n"
        "  normal — routine business correspondence\n\n"
        "Rules:\n"
        "  • Emails from executives (CEO, VP, Director) are usually urgent.\n"
        "  • Suspicious links, prize offers, or impersonation attempts are spam.\n"
        "  • Legal notices and security alerts are urgent.\n\n"
        "Respond with ONLY valid JSON — no markdown, no explanation:\n"
        '  {{"category": "<spam|urgent|normal>"}}\n\n'
        "Subject: {subject}\n\n"
        "Body:\n{body}"
    ),
    TaskName.MEDIUM.value: (
        "You are an expert email prioritisation assistant.\n\n"
        "PRIORITIES — choose exactly one:\n"
        "  low    — informational; no action needed within 2 days\n"
        "  medium — requires action within 1–2 business days\n"
        "  high   — requires immediate or same-day action\n\n"
        "Rules:\n"
        "  • Emails from the CEO, VP, Director, or Board are always HIGH.\n"
        "  • Outages, security incidents, legal threats, and data breaches are always HIGH.\n"
        "  • Spam and newsletters are always LOW.\n\n"
        "Respond with ONLY valid JSON — no markdown, no explanation:\n"
        '  {{"priority": "<low|medium|high>"}}\n\n'
        "Subject: {subject}\n\n"
        "Body:\n{body}"
    ),
    TaskName.HARD.value: (
        "You are a professional email triage assistant. Perform a complete 4-step triage.\n\n"
        "STEP 1 — Classify:  spam | urgent | normal\n"
        "STEP 2 — Priority:  low | medium | high\n"
        "STEP 3 — Action:\n"
        "  reply    — send an acknowledgement or response\n"
        "  escalate — forward to a senior person (use for urgent/legal/VIP/security issues)\n"
        "  ignore   — no response needed (spam, newsletters, FYI notices)\n"
        "STEP 4 — Response: 2–4 sentence professional reply.\n"
        "  • For 'escalate': acknowledge receipt and state it has been escalated.\n"
        "  • For 'reply': address the sender's request directly.\n"
        "  • For 'ignore': set response to empty string \"\".\n\n"
        "Intelligence rules:\n"
        "  • CEO/VP/Director/Board sender → HIGH priority + escalate.\n"
        "  • Suspicious link or prize → spam + ignore.\n"
        "  • Legal notice, security breach → urgent + escalate.\n\n"
        "Respond with ONLY this JSON — no markdown, no extra text:\n"
        "{{\n"
        '  "category": "<spam|urgent|normal>",\n'
        '  "priority": "<low|medium|high>",\n'
        '  "action": "<reply|escalate|ignore>",\n'
        '  "response": "<2-4 sentence reply, or empty string>"\n'
        "}}\n\n"
        "Subject: {subject}\n\n"
        "Body:\n{body}"
    ),
}

MAX_STEPS: Dict[str, int] = {
    TaskName.EASY.value:   1,
    TaskName.MEDIUM.value: 1,
    TaskName.HARD.value:   1,
}

ALL_TASKS: List[str] = [t.value for t in TaskName]


# ── Public helpers ─────────────────────────────────────────────────────────────

def sample_email(task_name: str, seed: Optional[int] = None) -> Email:
    """
    Return a reproducible random email for the given task + seed pair.
    Same (task_name, seed) always returns the same email.
    """
    if task_name not in MAX_STEPS:
        raise ValueError(f"Unknown task '{task_name}'. Valid: {ALL_TASKS}")

    effective_seed: Optional[int] = None
    if seed is not None:
        # XOR with hash to get different emails per task at the same seed
        effective_seed = seed ^ (hash(task_name) & 0xFFFFFFFF)

    rng = random.Random(effective_seed)
    return rng.choice(EMAILS)


def build_prompt(task_name: str, email: Email) -> str:
    """Fill the prompt template for the given task and email."""
    template = TASK_PROMPTS[task_name]
    return template.format(subject=email.subject, body=email.body)
