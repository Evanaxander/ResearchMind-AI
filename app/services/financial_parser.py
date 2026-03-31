"""
Financial Document Parser
--------------------------
Structure-aware parser that preserves financial tables, 
detects document types, and extracts metadata.

Unlike standard pypdf which destroys table formatting,
pdfplumber keeps tables intact as structured data.
This is critical for financial documents where a revenue
table split across two chunks becomes meaningless.
"""

import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import pdfplumber
from langchain.schema import Document


# ── Document type detection ───────────────────────────────────────────────────

DOCUMENT_TYPES = {
    "policy":     ["policy", "standard operating procedure", "sop", "guideline"],
    "contract":   ["agreement", "contract", "terms and conditions", "msa"],
    "technical":  ["architecture", "api", "system design", "technical specification"],
    "research":   ["methodology", "literature review", "findings", "abstract"],
    "report":     ["report", "executive summary", "conclusion", "appendix"],
    "10-K":       ["annual report", "form 10-k", "10-k filing"],
    "10-Q":       ["quarterly report", "form 10-q", "10-q filing"],
    "earnings":   ["earnings call", "conference call", "q1 ", "q2 ", "q3 ", "q4 "],
    "analyst":    ["price target", "buy rating", "sell rating", "initiating coverage"],
    "prospectus": ["prospectus", "ipo", "initial public offering"],
    "general":    [],
}

FINANCIAL_METRICS = [
    "revenue", "net income", "gross profit", "operating income",
    "ebitda", "eps", "earnings per share", "free cash flow",
    "total assets", "total liabilities", "debt", "equity",
    "guidance", "outlook", "margin", "return on equity",
]

GENERAL_SIGNALS = [
    "summary", "objective", "scope", "methodology", "findings", "recommendation",
    "risk", "timeline", "budget", "governance", "compliance", "kpi", "roadmap",
]


@dataclass
class ParsedDocument:
    """
    Structured representation of a parsed financial document.
    Preserves both raw text and extracted table data.
    """
    raw_text:      str
    tables:        list[dict]          = field(default_factory=list)
    doc_type:      str                 = "general"
    ticker:        Optional[str]       = None
    fiscal_period: Optional[str]       = None
    metrics_found: list[str]           = field(default_factory=list)
    page_count:    int                 = 0
    chunks:        list[Document]      = field(default_factory=list)


# ── Main parser class ─────────────────────────────────────────────────────────

class FinancialParser:
    """
    Parses financial documents with structure awareness.

    Key difference from standard RAG parsers:
    - Detects document type (10-K, earnings call, etc.)
    - Extracts tables as structured data, not flat text
    - Identifies which financial metrics are present
    - Creates finance-aware chunks that don't split tables
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        domain_mode: str = "general",
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.domain_mode   = domain_mode.lower().strip()

    def parse(self, file_path: Path, filename: str) -> ParsedDocument:
        """
        Main entry point. Parses a financial document and returns
        a structured ParsedDocument with text, tables, and metadata.
        """
        ext = filename.lower().split(".")[-1]

        if ext == "pdf":
            return self._parse_pdf(file_path, filename)
        elif ext == "txt":
            return self._parse_txt(file_path, filename)
        elif ext == "docx":
            return self._parse_docx(file_path, filename)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ── PDF parsing ───────────────────────────────────────────────────────────

    def _parse_pdf(self, file_path: Path, filename: str) -> ParsedDocument:
        """
        Uses pdfplumber to extract text AND tables separately.
        Tables are converted to markdown format to preserve structure.
        """
        all_text   = []
        all_tables = []

        with pdfplumber.open(str(file_path)) as pdf:
            page_count = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages):
                # Extract tables first — before text extraction
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        md_table = self._table_to_markdown(table, page_num + 1)
                        all_tables.append({
                            "page":     page_num + 1,
                            "markdown": md_table,
                            "rows":     len(table),
                        })
                        # Add table to text as markdown so it gets embedded
                        all_text.append(f"\n[TABLE — Page {page_num + 1}]\n{md_table}\n")

                # Extract regular text
                text = page.extract_text()
                if text:
                    all_text.append(text)

        raw_text = "\n\n".join(all_text)

        doc = ParsedDocument(
            raw_text   = raw_text,
            tables     = all_tables,
            page_count = page_count,
        )

        # Enrich with detected metadata
        doc.doc_type      = self._detect_doc_type(raw_text, filename)
        if self.domain_mode == "finance":
            doc.ticker        = self._extract_ticker(raw_text, filename)
            doc.fiscal_period = self._extract_fiscal_period(raw_text)
        doc.metrics_found = self._find_metrics(raw_text)
        doc.chunks        = self._create_chunks(doc, filename)

        return doc

    # ── TXT parsing ───────────────────────────────────────────────────────────

    def _parse_txt(self, file_path: Path, filename: str) -> ParsedDocument:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        doc = ParsedDocument(raw_text=raw_text, page_count=1)
        doc.doc_type      = self._detect_doc_type(raw_text, filename)
        if self.domain_mode == "finance":
            doc.ticker        = self._extract_ticker(raw_text, filename)
            doc.fiscal_period = self._extract_fiscal_period(raw_text)
        doc.metrics_found = self._find_metrics(raw_text)
        doc.chunks        = self._create_chunks(doc, filename)

        return doc

    # ── DOCX parsing ─────────────────────────────────────────────────────────

    def _parse_docx(self, file_path: Path, filename: str) -> ParsedDocument:
        import docx2txt
        raw_text = docx2txt.process(str(file_path))

        doc = ParsedDocument(raw_text=raw_text, page_count=1)
        doc.doc_type      = self._detect_doc_type(raw_text, filename)
        if self.domain_mode == "finance":
            doc.ticker        = self._extract_ticker(raw_text, filename)
            doc.fiscal_period = self._extract_fiscal_period(raw_text)
        doc.metrics_found = self._find_metrics(raw_text)
        doc.chunks        = self._create_chunks(doc, filename)

        return doc

    # ── Finance-aware chunking ────────────────────────────────────────────────

    def _create_chunks(self, doc: ParsedDocument, filename: str) -> list[Document]:
        """
        Creates chunks that respect financial document structure.

        Key rules:
        1. Never split a table across chunks
        2. Keep section headers with their content
        3. Add rich metadata to every chunk
        """
        chunks   = []
        sections = self._split_into_sections(doc.raw_text)

        for section_idx, section in enumerate(sections):
            # If section fits in one chunk, keep it whole
            if len(section) <= self.chunk_size:
                chunks.append(Document(
                    page_content = section.strip(),
                    metadata     = self._build_metadata(
                        doc, filename, section_idx, len(chunks)
                    ),
                ))
            else:
                # Split large sections with overlap
                sub_chunks = self._sliding_window(section)
                for i, sub in enumerate(sub_chunks):
                    chunks.append(Document(
                        page_content = sub.strip(),
                        metadata     = self._build_metadata(
                            doc, filename, section_idx, len(chunks)
                        ),
                    ))

        return chunks

    def _split_into_sections(self, text: str) -> list[str]:
        """
        Split on natural financial document section boundaries.
        Keeps tables intact — a [TABLE] block is never split.
        """
        # Protect tables from being split
        parts   = re.split(r'(\[TABLE[^\]]*\][^\[]+)', text)
        sections = []
        buffer  = ""

        for part in parts:
            if part.startswith("[TABLE"):
                # Flush buffer before table
                if buffer.strip():
                    sections.append(buffer.strip())
                    buffer = ""
                # Table is always its own section
                sections.append(part.strip())
            else:
                # Split on section headings
                sub_parts = re.split(
                    r'\n(?=[A-Z][A-Z\s]{5,50}\n)', part
                )
                for sp in sub_parts:
                    if len(buffer) + len(sp) > self.chunk_size:
                        if buffer.strip():
                            sections.append(buffer.strip())
                        buffer = sp
                    else:
                        buffer += "\n" + sp

        if buffer.strip():
            sections.append(buffer.strip())

        return [s for s in sections if len(s.strip()) > 50]

    def _sliding_window(self, text: str) -> list[str]:
        """Standard sliding window chunking for large text sections."""
        chunks = []
        start  = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # ── Metadata builder ──────────────────────────────────────────────────────

    def _build_metadata(
        self,
        doc: ParsedDocument,
        filename: str,
        section_idx: int,
        chunk_idx: int,
    ) -> dict:
        return {
            "filename":      filename,
            "doc_type":      doc.doc_type,
            "ticker":        doc.ticker or "unknown",
            "fiscal_period": doc.fiscal_period or "unknown",
            "metrics_found": ",".join(doc.metrics_found[:5]),
            "has_tables":    len(doc.tables) > 0,
            "domain_mode":   self.domain_mode,
            "section_idx":   section_idx,
            "chunk_index":   chunk_idx,
        }

    # ── Detection helpers ─────────────────────────────────────────────────────

    def _detect_doc_type(self, text: str, filename: str) -> str:
        combined = (text[:2000] + filename).lower()
        for doc_type, keywords in DOCUMENT_TYPES.items():
            if any(kw in combined for kw in keywords):
                return doc_type
        return "general"

    def _extract_ticker(self, text: str, filename: str) -> Optional[str]:
        # Common false positives to ignore
        FALSE_POSITIVES = {
            "LLC", "INC", "CORP", "LTD", "SEC", "USA", "CEO", "CFO",
            "CTO", "THE", "AND", "FOR", "ACT", "NET", "TAX", "EPS",
            "PDF", "FORM", "ITEM", "PART", "NOTE", "NONE", "NULL"
        }

        # First check filename for ticker hint e.g. "AAPL_10K.pdf"
        filename_upper = filename.upper()
        for known in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
                    "NVDA", "JPM", "GS", "MS", "BAC", "WFC"]:
            if known in filename_upper:
                return known

        # Search document text
        matches = re.findall(r'\b([A-Z]{2,5})\b', text[:3000])
        candidates = [
            m for m in matches
            if 2 <= len(m) <= 5 and m not in FALSE_POSITIVES
        ]

        if candidates:
            from collections import Counter
            return Counter(candidates).most_common(1)[0][0]
        return None


    def _extract_fiscal_period(self, text: str) -> Optional[str]:
        """Extract fiscal year or quarter from document text."""
        patterns = [
            r'fiscal year (\d{4})',
            r'year ended (?:december|january|march|june|september) \d+,? (\d{4})',
            r'(q[1-4] \d{4})',
            r'(\d{4}) annual report',
            r'for the year (\d{4})',
        ]
        text_lower = text[:3000].lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).upper()
        return None

    def _find_metrics(self, text: str) -> list[str]:
        """Return key signal terms based on the active domain mode."""
        text_lower = text.lower()
        terms = FINANCIAL_METRICS if self.domain_mode == "finance" else GENERAL_SIGNALS
        return [m for m in terms if m in text_lower]

    def _table_to_markdown(self, table: list[list], page_num: int) -> str:
        """
        Convert a pdfplumber table (list of rows) to markdown format.
        This preserves the structure so the LLM can read it correctly.
        """
        if not table:
            return ""

        rows = []
        for row in table:
            # Clean None values
            cleaned = [str(cell).strip() if cell else "" for cell in row]
            rows.append("| " + " | ".join(cleaned) + " |")

        if len(rows) > 1:
            # Add markdown header separator after first row
            cols      = len(table[0])
            separator = "| " + " | ".join(["---"] * cols) + " |"
            rows.insert(1, separator)

        return "\n".join(rows)