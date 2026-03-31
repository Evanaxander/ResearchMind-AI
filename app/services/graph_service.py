"""
Graph Service
--------------
Manages the Neo4j knowledge graph for FinanceIQ.

Every uploaded document becomes a node in the graph.
Relationships are automatically detected between documents:
  - SAME_COMPANY   → two documents about the same ticker
  - SAME_PERIOD    → two documents from the same fiscal period
  - REFERENCES     → one document mentions another
  - CONTRADICTS    → statements conflict between documents
  - UPDATES        → newer document supersedes an older one

This graph layer is what separates FinanceIQ from generic RAG.
When answering a question, the system doesn't just search FAISS —
it also traverses the graph to find related documents the user
didn't explicitly mention.
"""

from neo4j import GraphDatabase
from datetime import datetime
from typing import Optional
from app.core.config import settings


class GraphService:
    """
    Handles all Neo4j operations for FinanceIQ.

    Key operations:
    - add_document()        → creates a document node
    - link_related()        → finds and creates relationships
    - find_related_docs()   → traverses graph for related documents
    - detect_contradictions() → finds conflicting statements
    - get_document_graph()  → returns full graph for visualization
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=("neo4j", settings.NEO4J_PASSWORD),
        )
        self._create_indexes()

    def close(self):
        self.driver.close()

    # ── Schema setup ──────────────────────────────────────────────────────────

    def _create_indexes(self):
        """
        Create Neo4j indexes on first run.
        Indexes make graph queries fast — without them
        every query would scan the entire graph.
        """
        with self.driver.session() as session:
            session.run("""
                CREATE INDEX document_id IF NOT EXISTS
                FOR (d:Document) ON (d.doc_id)
            """)
            session.run("""
                CREATE INDEX document_ticker IF NOT EXISTS
                FOR (d:Document) ON (d.ticker)
            """)

    # ── Document node operations ──────────────────────────────────────────────

    def add_document(
        self,
        doc_id:        str,
        filename:      str,
        doc_type:      str,
        ticker:        Optional[str],
        fiscal_period: Optional[str],
        metrics_found: list[str],
        has_tables:    bool,
        chunk_count:   int,
    ):
        """
        Creates a Document node in Neo4j.
        Called automatically when a document is uploaded.

        After creating the node, automatically scans for
        relationships with existing documents.
        """
        with self.driver.session() as session:
            session.run("""
                MERGE (d:Document {doc_id: $doc_id})
                SET d.filename      = $filename,
                    d.doc_type      = $doc_type,
                    d.ticker        = $ticker,
                    d.fiscal_period = $fiscal_period,
                    d.metrics_found = $metrics_found,
                    d.has_tables    = $has_tables,
                    d.chunk_count   = $chunk_count,
                    d.created_at    = $created_at
            """, {
                "doc_id":        doc_id,
                "filename":      filename,
                "doc_type":      doc_type,
                "ticker":        ticker or "unknown",
                "fiscal_period": fiscal_period or "unknown",
                "metrics_found": metrics_found,
                "has_tables":    has_tables,
                "chunk_count":   chunk_count,
                "created_at":    datetime.utcnow().isoformat(),
            })

        # Auto-detect relationships with existing documents
        self.link_related(doc_id, ticker, fiscal_period, doc_type)

    def delete_document(self, doc_id: str):
        """Remove a document node and all its relationships."""
        with self.driver.session() as session:
            session.run("""
                MATCH (d:Document {doc_id: $doc_id})
                DETACH DELETE d
            """, {"doc_id": doc_id})

    # ── Relationship detection ────────────────────────────────────────────────

    def link_related(
        self,
        doc_id:        str,
        ticker:        Optional[str],
        fiscal_period: Optional[str],
        doc_type:      str,
    ):
        """
        Automatically detects and creates relationships
        between the new document and existing documents.

        This runs every time a document is uploaded —
        the graph grows smarter with every new document.
        """
        if ticker and ticker != "unknown":
            self._link_same_company(doc_id, ticker)

        if fiscal_period and fiscal_period != "unknown":
            self._link_same_period(doc_id, fiscal_period)

        self._link_updates(doc_id, ticker, doc_type)

    def _link_same_company(self, doc_id: str, ticker: str):
        """
        Links documents about the same company.
        Example: Apple's 10-K and Apple's earnings call
        both get a SAME_COMPANY relationship.
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Document {doc_id: $doc_id})
                MATCH (b:Document)
                WHERE b.ticker = $ticker
                  AND b.doc_id <> $doc_id
                MERGE (a)-[:SAME_COMPANY {ticker: $ticker}]->(b)
                MERGE (b)-[:SAME_COMPANY {ticker: $ticker}]->(a)
            """, {"doc_id": doc_id, "ticker": ticker})

    def _link_same_period(self, doc_id: str, fiscal_period: str):
        """
        Links documents from the same fiscal period.
        Example: Two Q3 2024 reports from different sources.
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Document {doc_id: $doc_id})
                MATCH (b:Document)
                WHERE b.fiscal_period = $fiscal_period
                  AND b.doc_id <> $doc_id
                MERGE (a)-[:SAME_PERIOD {period: $fiscal_period}]->(b)
                MERGE (b)-[:SAME_PERIOD {period: $fiscal_period}]->(a)
            """, {"doc_id": doc_id, "fiscal_period": fiscal_period})

    def _link_updates(self, doc_id: str, ticker: Optional[str], doc_type: str):
        """
        Links documents where one updates another.
        Example: A 10-Q updates a previous 10-Q for the same company.
        """
        if not ticker or ticker == "unknown":
            return

        with self.driver.session() as session:
            session.run("""
                MATCH (a:Document {doc_id: $doc_id})
                MATCH (b:Document)
                WHERE b.ticker   = $ticker
                  AND b.doc_type = $doc_type
                  AND b.doc_id  <> $doc_id
                  AND b.created_at < a.created_at
                MERGE (a)-[:UPDATES]->(b)
            """, {"doc_id": doc_id, "ticker": ticker, "doc_type": doc_type})

    def add_contradiction(
        self,
        doc_id_a:    str,
        doc_id_b:    str,
        description: str,
        severity:    str = "MEDIUM",
    ):
        """
        Creates a CONTRADICTS relationship between two documents.
        Called by the contradiction agent when it finds conflicts.
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Document {doc_id: $doc_id_a})
                MATCH (b:Document {doc_id: $doc_id_b})
                MERGE (a)-[r:CONTRADICTS]->(b)
                SET r.description = $description,
                    r.severity    = $severity,
                    r.detected_at = $detected_at
            """, {
                "doc_id_a":    doc_id_a,
                "doc_id_b":    doc_id_b,
                "description": description,
                "severity":    severity,
                "detected_at": datetime.utcnow().isoformat(),
            })

    # ── Graph traversal ───────────────────────────────────────────────────────

    def find_related_docs(
        self,
        doc_id:    str,
        max_depth: int = 2,
    ) -> list[dict]:
        """
        Traverses the graph to find documents related to a given doc_id.
        Returns related documents up to max_depth hops away.

        This is used by the Researcher agent to expand its search
        beyond the explicitly uploaded documents.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Document {doc_id: $doc_id})
                MATCH (a)-[r*1..2]-(b:Document)
                WHERE b.doc_id <> $doc_id
                RETURN DISTINCT
                    b.doc_id        AS doc_id,
                    b.filename      AS filename,
                    b.doc_type      AS doc_type,
                    b.ticker        AS ticker,
                    b.fiscal_period AS fiscal_period,
                    type(r[0])      AS relationship
                LIMIT 10
            """, {"doc_id": doc_id})

            return [dict(record) for record in result]

    def find_docs_by_ticker(self, ticker: str) -> list[dict]:
        """Find all documents about a specific company."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document {ticker: $ticker})
                RETURN d.doc_id        AS doc_id,
                       d.filename      AS filename,
                       d.doc_type      AS doc_type,
                       d.fiscal_period AS fiscal_period,
                       d.has_tables    AS has_tables
                ORDER BY d.created_at DESC
            """, {"ticker": ticker})

            return [dict(record) for record in result]

    def find_contradictions(self, ticker: Optional[str] = None) -> list[dict]:
        """Find all detected contradictions, optionally filtered by ticker."""
        with self.driver.session() as session:
            if ticker:
                result = session.run("""
                    MATCH (a:Document {ticker: $ticker})-[r:CONTRADICTS]->(b:Document)
                    RETURN a.filename      AS source_doc,
                           b.filename      AS target_doc,
                           r.description   AS description,
                           r.severity      AS severity,
                           r.detected_at   AS detected_at
                    ORDER BY r.detected_at DESC
                """, {"ticker": ticker})
            else:
                result = session.run("""
                    MATCH (a:Document)-[r:CONTRADICTS]->(b:Document)
                    RETURN a.filename      AS source_doc,
                           b.filename      AS target_doc,
                           r.description   AS description,
                           r.severity      AS severity,
                           r.detected_at   AS detected_at
                    ORDER BY r.detected_at DESC
                """)

            return [dict(record) for record in result]

    def get_document_graph(self) -> dict:
        """
        Returns the full graph as nodes + edges for visualization.
        Used by the frontend dashboard in Week 4.
        """
        with self.driver.session() as session:
            nodes_result = session.run("""
                MATCH (d:Document)
                RETURN d.doc_id        AS id,
                       d.filename      AS label,
                       d.doc_type      AS doc_type,
                       d.ticker        AS ticker,
                       d.fiscal_period AS fiscal_period,
                       d.chunk_count   AS chunk_count
            """)

            edges_result = session.run("""
                MATCH (a:Document)-[r]->(b:Document)
                RETURN a.doc_id    AS source,
                       b.doc_id    AS target,
                       type(r)     AS relationship,
                       r.severity  AS severity
            """)

            nodes = [dict(record) for record in nodes_result]
            edges = [dict(record) for record in edges_result]

            return {
                "nodes": nodes,
                "edges": edges,
                "summary": {
                    "total_documents":     len(nodes),
                    "total_relationships": len(edges),
                    "contradictions":      sum(
                        1 for e in edges if e["relationship"] == "CONTRADICTS"
                    ),
                }
            }

    def get_stats(self) -> dict:
        """Quick stats for the health check endpoint."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT d) AS nodes,
                       count(DISTINCT r) AS relationships
            """)
            record = result.single()
            return {
                "nodes":         record["nodes"],
                "relationships": record["relationships"],
            }