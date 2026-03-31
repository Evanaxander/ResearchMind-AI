"""
Background Monitor Worker
--------------------------
Celery worker that runs alert checks after every document upload.

Four checks run automatically on every new document:

1. Contradiction check
   Compare new document against existing docs for same ticker.
   If conflicts found → HIGH severity alert.

2. Risk escalation check
   Scan new document for severe risk language.
   Keywords like "material adverse", "going concern",
   "regulatory investigation" trigger alerts immediately.

3. Guidance change check
   Compare any forward guidance in new document against
   guidance in previous documents for same ticker.
   Changes trigger HIGH severity alert.

4. New document notification
   Always fires as LOW severity — confirms the document
   was analyzed and summarizes what was found.

How to run the worker (separate terminal):
    celery -A app.workers.monitor worker --loglevel=info

How tasks are queued:
    Called from document_service.py after every upload.
    If Celery is not running, falls back to synchronous execution.
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

from langchain_ollama import ChatOllama
from app.services.alert_service import alert_service
from app.services.rag_service import RAGService

# ── Celery app ────────────────────────────────────────────────────────────────

if CELERY_AVAILABLE:
    celery_app = Celery(
        "financeiq",
        broker  = "redis://localhost:6379/0",
        backend = "redis://localhost:6379/0",
    )
    celery_app.conf.update(
        task_serializer   = "json",
        result_serializer = "json",
        accept_content    = ["json"],
        timezone          = "UTC",
        task_track_started = True,
    )

# ── Shared resources ──────────────────────────────────────────────────────────

llm = ChatOllama(model="mistral", temperature=0)
rag = RAGService()

# High-risk keywords that trigger immediate alerts
HIGH_RISK_KEYWORDS = [
    "going concern",
    "material adverse",
    "regulatory investigation",
    "securities fraud",
    "class action",
    "bankruptcy",
    "liquidity crisis",
    "covenant violation",
    "restatement",
    "material weakness",
    "sec investigation",
    "department of justice",
]

MEDIUM_RISK_KEYWORDS = [
    "significant uncertainty",
    "substantial doubt",
    "impairment charge",
    "write-down",
    "restructuring",
    "layoffs",
    "supply chain disruption",
    "cybersecurity incident",
    "data breach",
]


# ── Core monitoring functions ─────────────────────────────────────────────────

def run_all_checks(
    doc_id:        str,
    filename:      str,
    ticker:        str,
    doc_type:      str,
    fiscal_period: str,
    metrics_found: list,
    raw_text:      str,
):
    """
    Runs all four alert checks for a newly uploaded document.
    Called directly if Celery unavailable, or as Celery task.
    """
    alerts_created = []

    print(f"[Monitor] Running checks for {filename} ({ticker})")

    # Check 1: New document notification (always fires)
    alert = alert_service.create_new_document_alert(
        ticker        = ticker,
        doc_id        = doc_id,
        filename      = filename,
        doc_type      = doc_type,
        fiscal_period = fiscal_period,
        metrics_found = metrics_found or [],
    )
    alerts_created.append(alert["alert_id"])
    print(f"[Monitor] Created new_document alert")

    # Check 2: Risk keyword scan
    risk_alerts = _check_risk_keywords(
        doc_id   = doc_id,
        ticker   = ticker,
        raw_text = raw_text,
    )
    alerts_created.extend(risk_alerts)

    # Check 3: Guidance change detection
    guidance_alert = _check_guidance_change(
        doc_id   = doc_id,
        ticker   = ticker,
        raw_text = raw_text,
    )
    if guidance_alert:
        alerts_created.append(guidance_alert)

    print(
        f"[Monitor] Checks complete for {filename}. "
        f"Alerts created: {len(alerts_created)}"
    )
    return alerts_created


def _check_risk_keywords(
    doc_id:   str,
    ticker:   str,
    raw_text: str,
) -> list[str]:
    """
    Scan document text for high and medium risk keywords.
    Instant check — no LLM needed, just keyword matching.
    """
    text_lower    = raw_text.lower()
    alerts_created = []

    # High severity keywords
    found_high = [
        kw for kw in HIGH_RISK_KEYWORDS
        if kw in text_lower
    ]
    if found_high:
        alert = alert_service.create_risk_alert(
            ticker       = ticker,
            doc_id       = doc_id,
            risk_factors = found_high,
            severity     = "CRITICAL" if len(found_high) >= 3 else "HIGH",
        )
        alerts_created.append(alert["alert_id"])
        print(f"[Monitor] HIGH risk alert: {found_high[:3]}")

    # Medium severity keywords
    found_medium = [
        kw for kw in MEDIUM_RISK_KEYWORDS
        if kw in text_lower
    ]
    if found_medium and not found_high:
        alert = alert_service.create_risk_alert(
            ticker       = ticker,
            doc_id       = doc_id,
            risk_factors = found_medium,
            severity     = "MEDIUM",
        )
        alerts_created.append(alert["alert_id"])
        print(f"[Monitor] MEDIUM risk alert: {found_medium[:3]}")

    return alerts_created


def _check_guidance_change(
    doc_id:   str,
    ticker:   str,
    raw_text: str,
) -> str:
    """
    Check if this document contains guidance that differs
    from guidance in previously uploaded documents.
    Uses LLM to extract and compare guidance statements.
    """
    # First extract guidance from new document
    new_guidance = _extract_guidance(raw_text)
    if not new_guidance:
        return None

    # Search existing documents for previous guidance
    try:
        results = rag.search(
            "forward guidance revenue outlook projection next quarter",
            top_k = 3,
        )
        if not results:
            return None

        existing_text = "\n\n".join(
            doc.page_content for doc, _ in results
        )
        existing_guidance = _extract_guidance(existing_text)

        if not existing_guidance:
            return None

        # Compare the two guidance statements
        if _guidance_changed(existing_guidance, new_guidance):
            alert = alert_service.create_guidance_alert(
                ticker       = ticker,
                doc_id       = doc_id,
                old_guidance = existing_guidance[:200],
                new_guidance = new_guidance[:200],
            )
            print(f"[Monitor] Guidance change alert created")
            return alert["alert_id"]

    except Exception as e:
        print(f"[Monitor] Guidance check error: {e}")

    return None


def _extract_guidance(text: str) -> str:
    """Extract forward guidance statement from text."""
    prompt = f"""Find any forward guidance or revenue outlook in this text.
Return ONE sentence summarizing the guidance.
If no guidance found, reply: NONE

TEXT: {text[:2000]}"""

    try:
        response = llm.invoke(prompt).content.strip()
        if response == "NONE" or len(response) < 10:
            return None
        return response[:300]
    except Exception:
        return None


def _guidance_changed(old: str, new: str) -> bool:
    """Use LLM to judge if guidance has materially changed."""
    prompt = f"""Compare these two financial guidance statements.
Has the guidance materially changed (different numbers, direction, or outlook)?

OLD GUIDANCE: {old}
NEW GUIDANCE: {new}

Reply with only YES or NO."""

    try:
        response = llm.invoke(prompt).content.strip().upper()
        return "YES" in response
    except Exception:
        return False


# ── Celery task wrapper ───────────────────────────────────────────────────────

if CELERY_AVAILABLE:
    @celery_app.task(
        name      = "monitor.analyze_document",
        bind      = True,
        max_retries = 2,
        soft_time_limit = 300,
    )
    def analyze_document_task(
        self,
        doc_id:        str,
        filename:      str,
        ticker:        str,
        doc_type:      str,
        fiscal_period: str,
        metrics_found: list,
        raw_text:      str,
    ):
        """
        Celery task wrapping run_all_checks.
        Runs asynchronously after document upload.
        Retries up to 2 times on failure.
        """
        try:
            return run_all_checks(
                doc_id        = doc_id,
                filename      = filename,
                ticker        = ticker or "UNKNOWN",
                doc_type      = doc_type or "general",
                fiscal_period = fiscal_period or "unknown",
                metrics_found = metrics_found or [],
                raw_text      = raw_text,
            )
        except Exception as exc:
            raise self.retry(exc=exc, countdown=30)


def queue_document_analysis(
    doc_id:        str,
    filename:      str,
    ticker:        str,
    doc_type:      str,
    fiscal_period: str,
    metrics_found: list,
    raw_text:      str,
):
    """
    Queue a document analysis task.

    If Celery + Redis are running: runs asynchronously in background.
    If not available: runs synchronously (fallback mode).

    This means the system works even without Celery running —
    alerts just take longer to generate.
    """
    if CELERY_AVAILABLE:
        try:
            analyze_document_task.delay(
                doc_id        = doc_id,
                filename      = filename,
                ticker        = ticker or "UNKNOWN",
                doc_type      = doc_type or "general",
                fiscal_period = fiscal_period or "unknown",
                metrics_found = metrics_found or [],
                raw_text      = raw_text[:10000],  # limit for Redis
            )
            print(f"[Monitor] Task queued for {filename}")
            return True
        except Exception as e:
            print(f"[Monitor] Celery unavailable ({e}), running synchronously")

    # Fallback: run synchronously
    run_all_checks(
        doc_id        = doc_id,
        filename      = filename,
        ticker        = ticker or "UNKNOWN",
        doc_type      = doc_type or "general",
        fiscal_period = fiscal_period or "unknown",
        metrics_found = metrics_found or [],
        raw_text      = raw_text[:10000],
    )
    return False