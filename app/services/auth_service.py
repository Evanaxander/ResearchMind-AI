"""
Auth Service
-------------
Handles JWT authentication and role-based access control for FinanceIQ.

Four roles with different access levels:

  analyst          → full access, detailed answers, all source chunks
  portfolio_manager → summaries + risk analysis, filtered sources
  compliance       → contradiction reports + regulatory citations
  executive        → one-paragraph summaries only, no raw data

In a real enterprise deployment this would connect to Active Directory
or an SSO provider. For FinanceIQ we use JWT tokens with role claims.
"""

from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from pathlib import Path
import json

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY  = "financeiq-secret-key-change-in-production"
ALGORITHM   = "HS256"
TOKEN_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── Role definitions ──────────────────────────────────────────────────────────

ROLES = {
    "analyst": {
        "label":          "Research Analyst",
        "answer_depth":   "full",
        "max_chunks":     10,
        "show_raw_text":  True,
        "show_analysis":  True,
        "can_query_contradictions": True,
        "description": (
            "Full access to all document data. "
            "Detailed answers with complete source citations and raw chunk text."
        ),
    },
    "portfolio_manager": {
        "label":          "Portfolio Manager",
        "answer_depth":   "summary",
        "max_chunks":     5,
        "show_raw_text":  False,
        "show_analysis":  True,
        "can_query_contradictions": True,
        "description": (
            "Access to summaries and risk analysis. "
            "Source citations shown but raw chunk text hidden."
        ),
    },
    "compliance": {
        "label":          "Compliance Officer",
        "answer_depth":   "compliance",
        "max_chunks":     8,
        "show_raw_text":  True,
        "show_analysis":  True,
        "can_query_contradictions": True,
        "description": (
            "Full access with focus on contradictions and regulatory citations. "
            "Every answer includes compliance notes."
        ),
    },
    "executive": {
        "label":          "Executive",
        "answer_depth":   "brief",
        "max_chunks":     3,
        "show_raw_text":  False,
        "show_analysis":  False,
        "can_query_contradictions": False,
        "description": (
            "Executive summaries only. "
            "One-paragraph answers, no raw data, no technical details."
        ),
    },
}

# ── Simple file-based user store ──────────────────────────────────────────────
# In production: replace with PostgreSQL
USERS_FILE = Path("./users.json")


def _load_users() -> dict:
    if USERS_FILE.exists():
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── User operations ───────────────────────────────────────────────────────────

def create_user(
    username: str,
    password: str,
    role:     str,
    email:    str = "",
) -> dict:
    """Create a new user. Raises ValueError if username exists or role invalid."""
    if role not in ROLES:
        raise ValueError(
            f"Invalid role '{role}'. Must be one of: {list(ROLES.keys())}"
        )

    users = _load_users()
    if username in users:
        raise ValueError(f"Username '{username}' already exists.")

    users[username] = {
        "username":      username,
        "email":         email,
        "role":          role,
        "password_hash": hash_password(password),
        "created_at":    datetime.utcnow().isoformat(),
        "last_login":    None,
    }
    _save_users(users)

    return {
        "username": username,
        "role":     role,
        "email":    email,
    }


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Verify credentials. Returns user dict or None if invalid."""
    users = _load_users()
    user  = users.get(username)

    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None

    # Update last login
    users[username]["last_login"] = datetime.utcnow().isoformat()
    _save_users(users)

    return user


def get_user(username: str) -> Optional[dict]:
    users = _load_users()
    return users.get(username)


# ── JWT token operations ──────────────────────────────────────────────────────

def create_token(username: str, role: str) -> str:
    """Create a JWT token with username and role claims."""
    now = datetime.utcnow()
    payload = {
        "sub":  username,
        "role": role,
        "exp":  now + timedelta(hours=TOKEN_HOURS),
        "iat":  int(now.timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token. Returns payload or None."""
    try:
        if token.startswith("Bearer "):
            token = token[7:]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# ── Role-based answer shaping ─────────────────────────────────────────────────

def shape_answer_for_role(
    role:        str,
    answer:      str,
    analysis:    str,
    sources:     list,
    agent_steps: list,
) -> dict:
    """
    Reshapes the query response based on the user's role.

    This is the core of role-based access:
    - Executives get a brief one-paragraph summary
    - Compliance officers get regulatory framing added
    - Analysts get everything including raw chunk text
    - Portfolio managers get summaries without raw data

    Same underlying data, different presentation per role.
    """
    role_config = ROLES.get(role, ROLES["analyst"])

    # Limit sources based on role
    shaped_sources = sources[:role_config["max_chunks"]]

    # Hide raw text for roles that shouldn't see it
    if not role_config["show_raw_text"]:
        shaped_sources = [
            {**s, "text": "[Raw text hidden for this role]"}
            if isinstance(s, dict)
            else s
            for s in shaped_sources
        ]

    # Shape the answer based on depth
    shaped_answer = answer
    shaped_analysis = analysis if role_config["show_analysis"] else None

    if role == "executive":
        # Summarize to one paragraph for executives
        shaped_answer = _summarize_for_executive(answer)
        shaped_analysis = None
        agent_steps    = ["[Executive view — agent steps hidden]"]

    elif role == "compliance":
        # Add compliance framing
        shaped_answer = (
            f"{answer}\n\n"
            f"[COMPLIANCE NOTE] This response is based on uploaded financial "
            f"documents. Always verify against primary source filings before "
            f"making compliance decisions."
        )

    return {
        "answer":      shaped_answer,
        "analysis":    shaped_analysis,
        "sources":     shaped_sources,
        "agent_steps": agent_steps,
        "role_label":  role_config["label"],
        "answer_depth": role_config["answer_depth"],
    }


def _summarize_for_executive(answer: str) -> str:
    """
    Trim a detailed answer to executive length.
    Takes the first paragraph or first 300 characters.
    """
    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
    if paragraphs:
        first = paragraphs[0]
        if len(first) <= 400:
            return first
        return first[:400] + "..."
    return answer[:400] + "..." if len(answer) > 400 else answer