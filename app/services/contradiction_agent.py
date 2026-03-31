"""
Contradiction Agent
--------------------
Detects contradictions between financial documents.

This is one of FinanceIQ's most unique features.
In finance, contradictions between documents matter enormously:
  - CEO says revenue will grow 15% in earnings call
  - 10-K risk factors say "significant revenue uncertainty"
  - Analyst report projects only 8% growth

These contradictions are signals that analysts spend days finding manually.
FinanceIQ finds them automatically.

How it works:
1. When a new document is uploaded, compare its key statements
   against statements from related documents (same ticker)
2. Use Mistral to judge whether statements contradict each other
3. Store confirmed contradictions as CONTRADICTS edges in Neo4j
4. Surface them in query responses and proactive alerts
"""

from langchain_ollama import ChatOllama
from app.services.graph_service import GraphService
from app.services.rag_service import RAGService


llm = ChatOllama(model="mistral", temperature=0)
rag = RAGService()


class ContradictionAgent:
    """
    Detects contradictions between financial documents.

    Two modes:
    1. check_on_upload()  → runs when a new document is uploaded,
                            compares against existing related docs
    2. check_on_query()   → runs during a query when the user asks
                            about contradictions explicitly
    """

    def __init__(self):
        self.graph = GraphService()

    # ── Upload-time contradiction check ───────────────────────────────────────

    def check_on_upload(
        self,
        new_doc_id: str,
        new_doc_text: str,
        ticker: str,
    ) -> list[dict]:
        """
        Called after a new document is uploaded.
        Compares the new document against all existing documents
        for the same company (ticker) and stores any contradictions
        found in Neo4j.

        Returns list of contradictions found.
        """
        if not ticker or ticker == "unknown":
            return []

        # Find all existing documents for this ticker
        related = self.graph.find_docs_by_ticker(ticker)
        related = [r for r in related if r["doc_id"] != new_doc_id]

        if not related:
            return []

        contradictions_found = []

        # Extract key statements from the new document
        new_statements = self._extract_key_statements(new_doc_text)
        if not new_statements:
            return []

        for related_doc in related[:3]:  # Check top 3 related docs
            related_doc_id = related_doc["doc_id"]

            # Get chunks from the related document
            related_chunks = self._get_doc_chunks(related_doc_id)
            if not related_chunks:
                continue

            related_text = "\n\n".join(related_chunks[:5])

            # Check for contradictions
            contradiction = self._detect_contradiction(
                text_a    = new_statements,
                text_b    = related_text,
                label_a   = "New document",
                label_b   = related_doc["filename"],
            )

            if contradiction["found"]:
                # Store in Neo4j graph
                self.graph.add_contradiction(
                    doc_id_a    = new_doc_id,
                    doc_id_b    = related_doc_id,
                    description = contradiction["description"],
                    severity    = contradiction["severity"],
                )
                contradictions_found.append({
                    "doc_a":       new_doc_id,
                    "doc_b":       related_doc_id,
                    "filename_b":  related_doc["filename"],
                    "description": contradiction["description"],
                    "severity":    contradiction["severity"],
                })

        return contradictions_found

    # ── Query-time contradiction check ────────────────────────────────────────

    def check_on_query(
        self,
        question: str,
        ticker:   str,
    ) -> str:
        """
        Called when a user explicitly asks about contradictions.
        Example questions:
          "Are there any contradictions in Apple's filings?"
          "Does the earnings call match the 10-K?"

        Returns a formatted contradiction report.
        """
        contradictions = self.graph.find_contradictions(ticker=ticker)

        if not contradictions:
            return f"No contradictions detected in documents for {ticker}."

        lines = [f"Found {len(contradictions)} contradiction(s) for {ticker}:\n"]

        for i, c in enumerate(contradictions, 1):
            lines.append(
                f"{i}. [{c['severity']}] {c['source_doc']} vs {c['target_doc']}\n"
                f"   {c['description']}\n"
                f"   Detected: {c['detected_at']}\n"
            )

        return "\n".join(lines)

    # ── Core contradiction detection ──────────────────────────────────────────

    def _detect_contradiction(
        self,
        text_a:  str,
        text_b:  str,
        label_a: str,
        label_b: str,
    ) -> dict:
        """
        Uses Mistral to judge whether two texts contradict each other.

        Returns:
          found       → True if contradiction detected
          description → plain English description of the contradiction
          severity    → HIGH, MEDIUM, or LOW
        """
        prompt = f"""You are a financial compliance analyst.
Compare these two financial document excerpts and determine if they contradict each other.

A contradiction means they make conflicting claims about the same topic —
for example, one says revenue will grow while the other says it will decline,
or one claims a risk is manageable while the other flags it as severe.

DOCUMENT A ({label_a}):
{text_a[:1500]}

DOCUMENT B ({label_b}):
{text_b[:1500]}

Respond in this exact format:
CONTRADICTION: YES or NO
SEVERITY: HIGH, MEDIUM, or LOW (only if YES)
DESCRIPTION: one sentence describing the contradiction (only if YES)

If no contradiction, just respond:
CONTRADICTION: NO"""

        try:
            response = llm.invoke(prompt).content.strip()
            return self._parse_contradiction_response(response)
        except Exception as e:
            return {"found": False, "description": "", "severity": "LOW"}

    def _parse_contradiction_response(self, response: str) -> dict:
        """Parse the structured response from the contradiction check."""
        lines      = response.strip().split("\n")
        found      = False
        severity   = "MEDIUM"
        description = ""

        for line in lines:
            line = line.strip()
            if line.startswith("CONTRADICTION:"):
                found = "YES" in line.upper()
            elif line.startswith("SEVERITY:"):
                for level in ["HIGH", "MEDIUM", "LOW"]:
                    if level in line.upper():
                        severity = level
                        break
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()

        return {
            "found":       found,
            "description": description,
            "severity":    severity,
        }

    # ── Helper methods ────────────────────────────────────────────────────────

    def _extract_key_statements(self, text: str) -> str:
        """
        Extract the most important financial statements
        from a document for contradiction checking.
        Focuses on guidance, risk factors, and metric claims.
        """
        prompt = f"""Extract the 5 most important financial claims or statements
from this document. Focus on:
- Revenue or earnings guidance
- Risk factor assessments
- Growth projections
- Market outlook statements

Return as a numbered list, one statement per line.

TEXT:
{text[:3000]}"""

        try:
            response = llm.invoke(prompt).content.strip()
            return response
        except Exception:
            return text[:1000]

    def _get_doc_chunks(self, doc_id: str) -> list[str]:
        """Retrieve top chunks for a document from FAISS."""
        try:
            results = rag.search(
                "financial results revenue earnings guidance risk",
                doc_ids=[doc_id],
                top_k=5,
            )
            return [doc.page_content for doc, _ in results]
        except Exception:
            return []