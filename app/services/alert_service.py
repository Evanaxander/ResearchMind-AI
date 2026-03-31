"""
Alert Service
--------------
Manages proactive alerts for FinanceIQ.

Alerts are generated automatically when:
1. A contradiction is detected between documents
2. A new document contains severe risk factors
3. Guidance changes significantly vs previous period
4. A financial metric looks anomalous

Alerts are stored in alerts.jsonl (append-only).
In production this would write to PostgreSQL + send emails/Slack.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

ALERTS_PATH = Path("./alerts.jsonl")

# ── Alert severity levels ─────────────────────────────────────────────────────

SEVERITY_LEVELS = {
    "CRITICAL": 4,   # Immediate action required
    "HIGH":     3,   # Review within 24 hours
    "MEDIUM":   2,   # Review within 1 week
    "LOW":      1,   # Informational
}

# ── Alert types ───────────────────────────────────────────────────────────────

ALERT_TYPES = {
    "contradiction":    "Contradiction Detected",
    "risk_escalation":  "Risk Factor Escalation",
    "guidance_change":  "Guidance Change",
    "metric_anomaly":   "Metric Anomaly",
    "new_document":     "New Document Analyzed",
}


class AlertService:
    """
    Creates, stores, and retrieves FinanceIQ alerts.

    Every alert has:
    - alert_id    → unique identifier
    - alert_type  → what triggered it
    - severity    → CRITICAL/HIGH/MEDIUM/LOW
    - ticker      → which company
    - title       → short headline
    - description → full explanation
    - doc_ids     → which documents are involved
    - created_at  → when detected
    - read        → whether the user has seen it
    """

    # ── Create alerts ─────────────────────────────────────────────────────────

    def create_alert(
        self,
        alert_type:  str,
        severity:    str,
        ticker:      str,
        title:       str,
        description: str,
        doc_ids:     list[str],
        metadata:    dict = None,
    ) -> dict:
        """Create and store a new alert."""
        alert = {
            "alert_id":   str(uuid.uuid4()),
            "alert_type": alert_type,
            "type_label": ALERT_TYPES.get(alert_type, alert_type),
            "severity":   severity,
            "ticker":     ticker,
            "title":      title,
            "description": description,
            "doc_ids":    doc_ids,
            "metadata":   metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "read":       False,
        }

        self._write(alert)
        return alert

    def create_contradiction_alert(
        self,
        ticker:      str,
        doc_a:       str,
        doc_b:       str,
        description: str,
        severity:    str = "HIGH",
    ) -> dict:
        return self.create_alert(
            alert_type  = "contradiction",
            severity    = severity,
            ticker      = ticker,
            title       = f"Contradiction detected in {ticker} documents",
            description = description,
            doc_ids     = [doc_a, doc_b],
            metadata    = {"doc_a": doc_a, "doc_b": doc_b},
        )

    def create_risk_alert(
        self,
        ticker:       str,
        doc_id:       str,
        risk_factors: list[str],
        severity:     str = "MEDIUM",
    ) -> dict:
        return self.create_alert(
            alert_type  = "risk_escalation",
            severity    = severity,
            ticker      = ticker,
            title       = f"New risk factors detected in {ticker} filing",
            description = (
                f"The newly uploaded document contains "
                f"{len(risk_factors)} risk factor(s) requiring attention: "
                f"{'; '.join(risk_factors[:2])}"
                f"{'...' if len(risk_factors) > 2 else '.'}"
            ),
            doc_ids     = [doc_id],
            metadata    = {"risk_factors": risk_factors},
        )

    def create_guidance_alert(
        self,
        ticker:     str,
        doc_id:     str,
        old_guidance: str,
        new_guidance: str,
    ) -> dict:
        return self.create_alert(
            alert_type  = "guidance_change",
            severity    = "HIGH",
            ticker      = ticker,
            title       = f"Guidance change detected for {ticker}",
            description = (
                f"Forward guidance has changed.\n"
                f"Previous: {old_guidance}\n"
                f"Current:  {new_guidance}"
            ),
            doc_ids     = [doc_id],
            metadata    = {
                "old_guidance": old_guidance,
                "new_guidance": new_guidance,
            },
        )

    def create_new_document_alert(
        self,
        ticker:       str,
        doc_id:       str,
        filename:     str,
        doc_type:     str,
        fiscal_period: str,
        metrics_found: list[str],
    ) -> dict:
        return self.create_alert(
            alert_type  = "new_document",
            severity    = "LOW",
            ticker      = ticker,
            title       = f"New {doc_type} uploaded for {ticker}",
            description = (
                f"'{filename}' ({fiscal_period}) has been analyzed. "
                f"Detected metrics: {', '.join(metrics_found[:5])}."
            ),
            doc_ids     = [doc_id],
        )

    # ── Read alerts ───────────────────────────────────────────────────────────

    def get_all(
        self,
        ticker:     Optional[str] = None,
        severity:   Optional[str] = None,
        unread_only: bool = False,
        limit:      int = 50,
    ) -> list[dict]:
        """
        Returns alerts filtered by ticker, severity, and read status.
        Most recent first.
        """
        alerts = self._load_all()

        if ticker:
            alerts = [a for a in alerts if a.get("ticker") == ticker.upper()]
        if severity:
            alerts = [a for a in alerts if a.get("severity") == severity.upper()]
        if unread_only:
            alerts = [a for a in alerts if not a.get("read", False)]

        # Sort by created_at descending
        alerts.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return alerts[:limit]

    def get_by_id(self, alert_id: str) -> Optional[dict]:
        alerts = self._load_all()
        for alert in alerts:
            if alert.get("alert_id") == alert_id:
                return alert
        return None

    def mark_read(self, alert_id: str) -> bool:
        """Mark an alert as read."""
        alerts = self._load_all()
        updated = False

        for alert in alerts:
            if alert.get("alert_id") == alert_id:
                alert["read"] = True
                updated = True
                break

        if updated:
            self._rewrite_all(alerts)
        return updated

    def mark_all_read(self, ticker: Optional[str] = None):
        """Mark all alerts as read, optionally filtered by ticker."""
        alerts = self._load_all()
        for alert in alerts:
            if ticker is None or alert.get("ticker") == ticker:
                alert["read"] = True
        self._rewrite_all(alerts)

    def get_stats(self) -> dict:
        alerts = self._load_all()
        if not alerts:
            return {
                "total": 0, "unread": 0,
                "by_severity": {}, "by_type": {},
            }

        unread      = sum(1 for a in alerts if not a.get("read", False))
        by_severity = {}
        by_type     = {}

        for alert in alerts:
            sev  = alert.get("severity", "UNKNOWN")
            typ  = alert.get("alert_type", "UNKNOWN")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_type[typ]     = by_type.get(typ, 0) + 1

        return {
            "total":       len(alerts),
            "unread":      unread,
            "by_severity": by_severity,
            "by_type":     by_type,
        }

    # ── Storage helpers ───────────────────────────────────────────────────────

    def _write(self, alert: dict):
        with open(ALERTS_PATH, "a") as f:
            f.write(json.dumps(alert) + "\n")

    def _load_all(self) -> list[dict]:
        if not ALERTS_PATH.exists():
            return []
        alerts = []
        with open(ALERTS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        alerts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return alerts

    def _rewrite_all(self, alerts: list[dict]):
        with open(ALERTS_PATH, "w") as f:
            for alert in alerts:
                f.write(json.dumps(alert) + "\n")


# Singleton
alert_service = AlertService()