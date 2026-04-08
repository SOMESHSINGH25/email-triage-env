from typing import Dict, List
from env.models import Email

# ---------------------------------------------------------------------------
# Synthetic email corpus
# ---------------------------------------------------------------------------

EMAILS: List[Dict] = [
    # --- spam ---
    {
        "id": "e001",
        "subject": "You've WON $1,000,000!!!",
        "body": "Congratulations! Click here to claim your prize now. Limited time offer!",
        "sender": "winner@totally-legit.biz",
        "timestamp": "2024-01-15T08:00:00Z",
    },
    {
        "id": "e002",
        "subject": "URGENT: Your account will be DELETED",
        "body": "Please send your password immediately or your account will be deleted within 24 hours.",
        "sender": "security@fake-bank.net",
        "timestamp": "2024-01-15T08:05:00Z",
    },
    # --- billing ---
    {
        "id": "e003",
        "subject": "Invoice #4821 — overdue by 30 days",
        "body": "Hello, I'm writing because invoice #4821 for $2,400 is now 30 days past due. Please process payment or contact us to discuss. This may affect your service.",
        "sender": "accounts@acme-corp.com",
        "timestamp": "2024-01-15T09:00:00Z",
    },
    {
        "id": "e004",
        "subject": "Incorrect charge on my account",
        "body": "I was billed $299 on Jan 10th but my plan is $99/month. Please investigate and refund the difference as soon as possible.",
        "sender": "customer.jane@gmail.com",
        "timestamp": "2024-01-15T09:30:00Z",
    },
    # --- support ---
    {
        "id": "e005",
        "subject": "Cannot log in to my account",
        "body": "Hi, I've been trying to log in for the past two days and keep getting an 'invalid credentials' error. I've already reset my password twice. Please help.",
        "sender": "bob.smith@outlook.com",
        "timestamp": "2024-01-15T10:00:00Z",
    },
    {
        "id": "e006",
        "subject": "App crashes on startup",
        "body": "Your mobile app crashes immediately on opening since the latest update (v3.2.1). Device: iPhone 14 Pro, iOS 17.2. This is very frustrating.",
        "sender": "alice.j@company.io",
        "timestamp": "2024-01-15T10:15:00Z",
    },
    # --- sales ---
    {
        "id": "e007",
        "subject": "Interested in enterprise plan",
        "body": "Hi, we're a team of 500 and currently evaluating your platform against competitors. Could someone from sales reach out to discuss pricing and volume discounts?",
        "sender": "procurement@bigcorp.com",
        "timestamp": "2024-01-15T11:00:00Z",
    },
    {
        "id": "e008",
        "subject": "Request for product demo",
        "body": "We've been looking at your analytics product. We'd love to see a live demo for our team next week if possible.",
        "sender": "cto@startup-x.io",
        "timestamp": "2024-01-15T11:30:00Z",
    },
    # --- engineering / bug reports ---
    {
        "id": "e009",
        "subject": "API rate limiting not working correctly",
        "body": "Your /v2/data endpoint is returning 429 errors even when we're well within our rate limits (10 req/s, we're doing 3 req/s). Reproducible 100% of the time. Here are the logs: [attached]",
        "sender": "devops@partner-co.com",
        "timestamp": "2024-01-15T12:00:00Z",
    },
    {
        "id": "e010",
        "subject": "Data export producing corrupted CSV",
        "body": "The bulk data export feature produces CSV files with garbled UTF-8 characters in the 'description' column. Affects all exports since Jan 12.",
        "sender": "dataeng@client-firm.com",
        "timestamp": "2024-01-15T12:30:00Z",
    },
]

# Ground-truth labels for graders
GROUND_TRUTH: Dict[str, Dict] = {
    "e001": {"category": "spam",        "urgency": "low",      "queue": "spam"},
    "e002": {"category": "spam",        "urgency": "low",      "queue": "spam"},
    "e003": {"category": "billing",     "urgency": "high",     "queue": "billing"},
    "e004": {"category": "billing",     "urgency": "medium",   "queue": "billing"},
    "e005": {"category": "support",     "urgency": "medium",   "queue": "support"},
    "e006": {"category": "support",     "urgency": "high",     "queue": "engineering"},
    "e007": {"category": "sales",       "urgency": "medium",   "queue": "sales"},
    "e008": {"category": "sales",       "urgency": "low",      "queue": "sales"},
    "e009": {"category": "engineering", "urgency": "critical", "queue": "engineering"},
    "e010": {"category": "engineering", "urgency": "high",     "queue": "engineering"},
}

# Reference replies for Task 3 grading
REFERENCE_REPLIES: Dict[str, Dict] = {
    "e005": {
        "reference": (
            "Hi Bob, thank you for reaching out. I'm sorry you're having trouble logging in. "
            "Our team is looking into this immediately. "
            "In the meantime, could you try clearing your browser cache and cookies? "
            "We'll follow up within 2 business hours with a resolution."
        ),
        "required_keywords": ["sorry", "looking into", "follow up", "hours"],
        "forbidden_phrases": ["your fault", "you should have", "not our problem"],
        "policy_rules": {
            "must_apologise": True,
            "must_give_eta": True,
            "no_blame": True,
        },
    },
    "e004": {
        "reference": (
            "Hello, thank you for bringing this to our attention. "
            "I've reviewed your account and can confirm the discrepancy. "
            "We will process a refund of $200 to your original payment method within 5-7 business days. "
            "Please don't hesitate to reach out if you have further questions."
        ),
        "required_keywords": ["refund", "business days", "account", "confirm"],
        "forbidden_phrases": ["cannot refund", "no refund", "your fault"],
        "policy_rules": {
            "must_apologise": False,
            "must_give_eta": True,
            "no_blame": True,
        },
    },
}

VALID_CATEGORIES = {"spam", "billing", "support", "sales", "engineering"}
VALID_URGENCIES  = {"low", "medium", "high", "critical"}
VALID_QUEUES     = {"spam", "billing", "support", "sales", "engineering"}

def get_emails_by_ids(ids: List[str]) -> List[Email]:
    index = {e["id"]: e for e in EMAILS}
    return [Email(**index[i]) for i in ids if i in index]

def all_emails() -> List[Email]:
    return [Email(**e) for e in EMAILS]