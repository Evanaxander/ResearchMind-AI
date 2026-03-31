"""
Audit Logger
-------------
Logs every query made to FinanceIQ with full traceability.

In financial services, audit trails are a regulatory requirement.
Every query must be logged with:
  - Who asked (username + role)
  - What they asked (the question)
  - When they asked (timestamp)
  - Which documents were searched (doc_ids)
  - How long it took (latency)
  - How many sources were returned

This log is stored in audit_log.jsonl (one JSON per line).
In production this would write to PostgreSQL or a SIEM system.

Why this matters for interviews:
"Financial institutions operate under strict regulatory frameworks.
MiFID II, SEC regulations, and internal compliance requirements
all mandate that research activities are fully auditable.
FinanceIQ logs every query so compliance teams can answer:
who accessed what information and when."
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

AUDIT_LOG_PATH = Path("./audit_log.jsonl")


class AuditLogger:
    """
    Append-only audit log for all FinanceIQ queries.

    Each log entry is a single line of JSON (JSONL format).
    JSONL is preferred over a single JSON array because:
    - Each line is independently parseable
    - New entries can be appended without reading the whole file
    - Easy to stream to log aggregation systems
    """

    def log_query(
        self,
        username:      str,
        role:          str,
        question:      str,
        doc_ids:       Optional[list],
        query_type:    Optional[str],
        sources_count: int,
        latency_ms:    Optional[float],
        success:       bool = True,
        error:         Optional[str] = None,
    ):
        """
        Log a query event.
        Called by the query router after every request.
        """
        entry = {
            "event":         "query",
            "timestamp":     datetime.utcnow().isoformat(),
            "username":      username,
            "role":          role,
            "question":      question[:200],   # truncate for log size
            "query_type":    query_type,
            "doc_ids":       doc_ids,
            "sources_count": sources_count,
            "latency_ms":    latency_ms,
            "success":       success,
            "error":         error,
        }
        self._write(entry)

    def log_upload(
        self,
        username:        str,
        role:            str,
        filename:        str,
        doc_type:        Optional[str],
        ticker:          Optional[str],
        chunk_count:     int,
        contradictions:  int,
    ):
        """Log a document upload event."""
        entry = {
            "event":            "upload",
            "timestamp":        datetime.utcnow().isoformat(),
            "username":         username,
            "role":             role,
            "filename":         filename,
            "doc_type":         doc_type,
            "ticker":           ticker,
            "chunk_count":      chunk_count,
            "contradictions":   contradictions,
        }
        self._write(entry)

    def log_login(self, username: str, role: str, success: bool):
        """Log a login event."""
        entry = {
            "event":     "login",
            "timestamp": datetime.utcnow().isoformat(),
            "username":  username,
            "role":      role,
            "success":   success,
        }
        self._write(entry)

    def get_recent(self, limit: int = 50) -> list[dict]:
        """
        Return the most recent audit log entries.
        Used by the compliance dashboard.
        """
        if not AUDIT_LOG_PATH.exists():
            return []

        entries = []
        with open(AUDIT_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return entries[-limit:]

    def get_user_history(self, username: str, limit: int = 20) -> list[dict]:
        """Return audit log entries for a specific user."""
        all_entries = self.get_recent(limit=1000)
        user_entries = [
            e for e in all_entries
            if e.get("username") == username
        ]
        return user_entries[-limit:]

    def get_stats(self) -> dict:
        """
        Aggregate stats from the audit log.
        Used by the compliance dashboard overview.
        """
        entries = self.get_recent(limit=1000)

        if not entries:
            return {
                "total_queries":  0,
                "total_uploads":  0,
                "unique_users":   0,
                "queries_by_role": {},
                "avg_latency_ms": None,
            }

        queries  = [e for e in entries if e["event"] == "query"]
        uploads  = [e for e in entries if e["event"] == "upload"]
        users    = set(e["username"] for e in entries)

        latencies = [
            e["latency_ms"] for e in queries
            if e.get("latency_ms") is not None
        ]

        role_counts = {}
        for q in queries:
            role = q.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1

        return {
            "total_queries":   len(queries),
            "total_uploads":   len(uploads),
            "unique_users":    len(users),
            "queries_by_role": role_counts,
            "avg_latency_ms":  round(
                sum(latencies) / len(latencies), 2
            ) if latencies else None,
        }

    def _write(self, entry: dict):
        """Append one JSON line to the audit log."""
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")


# Singleton instance used across the app
audit_logger = AuditLogger()