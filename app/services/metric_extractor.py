"""
Financial Metric Extractor
---------------------------
Automatically extracts key financial metrics from parsed documents.

When a document is uploaded, this runs in the background and pulls:
- Revenue, net income, gross margin
- EPS (earnings per share)
- Guidance and outlook statements
- Risk factor summaries

These extracted metrics are stored as document metadata,
enabling fast financial queries without re-reading entire documents.
"""

import re
from typing import Optional
from dataclasses import dataclass, field
from langchain_ollama import ChatOllama

from app.core.config import settings


@dataclass
class ExtractedMetrics:
    """
    Structured financial metrics extracted from a document.
    All fields are optional — not every document contains every metric.
    """
    ticker:           Optional[str]   = None
    fiscal_period:    Optional[str]   = None
    doc_type:         Optional[str]   = None

    # Income statement
    revenue:          Optional[str]   = None
    net_income:       Optional[str]   = None
    gross_margin:     Optional[str]   = None
    operating_income: Optional[str]   = None
    ebitda:           Optional[str]   = None
    eps:              Optional[str]   = None

    # Balance sheet
    total_assets:     Optional[str]   = None
    total_debt:       Optional[str]   = None

    # Guidance
    guidance:         Optional[str]   = None
    outlook:          Optional[str]   = None

    # Risk factors (top 3)
    risk_factors:     list[str]       = field(default_factory=list)

    # Raw extraction confidence
    confidence:       float           = 0.0
    # General-purpose enrichment
    key_topics:       list[str]       = field(default_factory=list)
    document_intent:  Optional[str]   = None


class MetricExtractor:
    """
    Uses Mistral (local via Ollama) to extract financial metrics
    from document text using structured prompts.

    Why LLM-based extraction instead of regex?
    Financial documents express the same number in many ways:
      - "Revenue of $94.9 billion"
      - "Net sales: $94,929 million"
      - "Total revenues increased 8% to $94.9B"
    Regex can't handle all variations. An LLM can.
    """

    def __init__(self, domain_mode: str = "general", enabled: bool = False):
        self.domain_mode = domain_mode.lower().strip()
        self.enabled = enabled
        self.llm = ChatOllama(model="mistral", temperature=0)

    def extract(self, text: str, doc_type: str = "general") -> ExtractedMetrics:
        """
        Extract all financial metrics from document text.
        Uses the first 6000 characters — metrics are usually in the summary.
        """
        sample = text[:6000]
        metrics = ExtractedMetrics(doc_type=doc_type)

        if not self.enabled:
            metrics.key_topics = self._keyword_topics(sample)
            metrics.document_intent = self._guess_intent(sample)
            return metrics

        if self.domain_mode != "finance":
            metrics.key_topics = self._extract_topics(sample)
            metrics.document_intent = self._extract_document_intent(sample)
            metrics.confidence = 0.75 if metrics.key_topics or metrics.document_intent else 0.4
            return metrics

        # Run extractions
        metrics.revenue          = self._extract_field(sample, "total revenue or net sales")
        metrics.net_income       = self._extract_field(sample, "net income or net earnings")
        metrics.gross_margin     = self._extract_field(sample, "gross margin percentage")
        metrics.eps              = self._extract_field(sample, "earnings per share (EPS)")
        metrics.ebitda           = self._extract_field(sample, "EBITDA")
        metrics.total_debt       = self._extract_field(sample, "total debt or long-term debt")
        metrics.guidance         = self._extract_guidance(sample)
        metrics.risk_factors     = self._extract_risk_factors(sample)
        metrics.confidence       = self._calculate_confidence(metrics)

        return metrics

    def extract_comparison(
        self,
        text_a: str,
        text_b: str,
        label_a: str,
        label_b: str,
    ) -> dict:
        """
        Compare metrics between two documents.
        Used by the comparative analysis agent.
        Example: Compare Apple Q3 2024 vs Q3 2023
        """
        metrics_a = self.extract(text_a)
        metrics_b = self.extract(text_b)

        comparison = {}
        fields = ["revenue", "net_income", "gross_margin", "eps", "ebitda"]

        for field_name in fields:
            val_a = getattr(metrics_a, field_name)
            val_b = getattr(metrics_b, field_name)

            if val_a or val_b:
                comparison[field_name] = {
                    label_a: val_a or "not found",
                    label_b: val_b or "not found",
                }

        return comparison

    # ── Private extraction methods ────────────────────────────────────────────

    def _extract_field(self, text: str, field_description: str) -> Optional[str]:
        """
        Ask Mistral to find one specific metric in the text.
        Returns the value as a string, or None if not found.
        """
        prompt = f"""You are a financial data extractor.
Find the {field_description} in the text below.
Reply with ONLY the value and its unit (e.g. "$94.9 billion" or "42.5%").
If not found, reply with exactly: NOT_FOUND

TEXT:
{text[:3000]}"""

        try:
            response = self.llm.invoke(prompt).content.strip()
            if "NOT_FOUND" in response or len(response) > 100:
                return None
            return response
        except Exception:
            return None

    def _extract_guidance(self, text: str) -> Optional[str]:
        """Extract forward guidance — management's prediction for next period."""
        prompt = f"""You are a financial analyst.
Find any forward guidance or outlook statements in the text.
These are predictions management makes about future revenue, earnings, or growth.
Summarize in ONE sentence. If none found, reply: NOT_FOUND

TEXT:
{text[:4000]}"""

        try:
            response = self.llm.invoke(prompt).content.strip()
            if "NOT_FOUND" in response:
                return None
            return response[:300]
        except Exception:
            return None

    def _extract_risk_factors(self, text: str) -> list[str]:
        """Extract the top 3 risk factors mentioned in the document."""
        prompt = f"""You are a financial risk analyst.
List the top 3 risk factors mentioned in this text.
Format as a numbered list, one per line, each under 20 words.
If fewer than 3 are found, list what you find.

TEXT:
{text[:4000]}"""

        try:
            response = self.llm.invoke(prompt).content.strip()
            risks = []
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Remove numbering
                line = re.sub(r'^[\d]+[.)]\s*', '', line).strip()
                if line and len(line) > 10:
                    risks.append(line)
            return risks[:3]
        except Exception:
            return []

    def _calculate_confidence(self, metrics: ExtractedMetrics) -> float:
        """
        Simple confidence score based on how many fields were extracted.
        1.0 = all key fields found, 0.0 = nothing found.
        """
        key_fields = [
            metrics.revenue, metrics.net_income,
            metrics.eps, metrics.guidance,
        ]
        found = sum(1 for f in key_fields if f is not None)
        return round(found / len(key_fields), 2)

    def _extract_topics(self, text: str) -> list[str]:
        prompt = f"""List up to 5 concise key topics covered in this document.
Return one topic per line with no numbering.

TEXT:
{text[:4000]}"""
        try:
            response = self.llm.invoke(prompt).content.strip()
            topics = []
            for line in response.split("\n"):
                clean = re.sub(r'^[\d]+[.)]\s*', '', line.strip())
                if clean and 2 <= len(clean) <= 80:
                    topics.append(clean)
            return topics[:5]
        except Exception:
            return self._keyword_topics(text)

    def _extract_document_intent(self, text: str) -> Optional[str]:
        prompt = f"""In one sentence, describe the primary purpose of this document.
If unclear, reply NOT_FOUND.

TEXT:
{text[:3500]}"""
        try:
            response = self.llm.invoke(prompt).content.strip()
            if "NOT_FOUND" in response:
                return None
            return response[:220]
        except Exception:
            return self._guess_intent(text)

    def _keyword_topics(self, text: str) -> list[str]:
        candidates = [
            "summary", "objective", "scope", "methodology", "findings", "recommendation",
            "risk", "compliance", "timeline", "budget", "governance", "kpi",
        ]
        lowered = text.lower()
        return [c for c in candidates if c in lowered][:5]

    def _guess_intent(self, text: str) -> Optional[str]:
        lowered = text.lower()
        if "recommend" in lowered or "recommendation" in lowered:
            return "Provide recommendations and supporting rationale."
        if "policy" in lowered or "compliance" in lowered:
            return "Define policy or compliance expectations."
        if "analysis" in lowered or "findings" in lowered:
            return "Present analysis findings and conclusions."
        return None

    def format_for_display(self, metrics: ExtractedMetrics) -> str:
        """Format extracted metrics as a readable summary string."""
        if self.domain_mode != "finance":
            lines = [f"Document type: {metrics.doc_type or 'general'}", ""]
            if metrics.document_intent:
                lines.append(f"Intent: {metrics.document_intent}")
            if metrics.key_topics:
                lines.append("Key topics:")
                for i, topic in enumerate(metrics.key_topics, 1):
                    lines.append(f"  {i}. {topic}")
            if len(lines) == 2:
                lines.append("No additional enrichment available.")
            return "\n".join(lines)

        lines = []

        if metrics.ticker:
            lines.append(f"Ticker: {metrics.ticker}")
        if metrics.fiscal_period:
            lines.append(f"Period: {metrics.fiscal_period}")
        if metrics.doc_type:
            lines.append(f"Document type: {metrics.doc_type}")

        lines.append("")

        if metrics.revenue:
            lines.append(f"Revenue: {metrics.revenue}")
        if metrics.net_income:
            lines.append(f"Net income: {metrics.net_income}")
        if metrics.gross_margin:
            lines.append(f"Gross margin: {metrics.gross_margin}")
        if metrics.eps:
            lines.append(f"EPS: {metrics.eps}")
        if metrics.ebitda:
            lines.append(f"EBITDA: {metrics.ebitda}")
        if metrics.total_debt:
            lines.append(f"Total debt: {metrics.total_debt}")

        if metrics.guidance:
            lines.append(f"\nGuidance: {metrics.guidance}")

        if metrics.risk_factors:
            lines.append("\nTop risk factors:")
            for i, risk in enumerate(metrics.risk_factors, 1):
                lines.append(f"  {i}. {risk}")

        lines.append(f"\nExtraction confidence: {metrics.confidence:.0%}")

        return "\n".join(lines)