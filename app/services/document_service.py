import json
import aiofiles
from pathlib import Path
from typing import Optional

from app.models.schemas import DocumentMetadata
from app.core.config import settings
from app.services.rag_service import RAGService
from app.services.financial_parser import FinancialParser
from app.services.metric_extractor import MetricExtractor
from app.services.graph_service import GraphService
from app.services.contradiction_agent import ContradictionAgent
from app.workers.monitor import queue_document_analysis


class DocumentService:
    """
    Full document pipeline for FinanceIQ.

    On every upload:
    1. Save file to disk
    2. Parse with FinancialParser (structure-aware, table-preserving)
    3. Extract financial metrics with MetricExtractor
    4. Index enriched chunks into FAISS
    5. Add document node to Neo4j graph
    6. Auto-detect relationships with existing documents
    7. Run contradiction check against related documents
    8. Queue proactive alert checks (Celery background worker)
    9. Return metadata + financial summary + contradictions
    """

    def __init__(self):
        self.upload_dir     = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.upload_dir / "_registry.json"
        self.rag            = RAGService()
        self.parser         = FinancialParser()
        self.extractor      = MetricExtractor()
        self.graph          = GraphService()
        self.contradiction  = ContradictionAgent()

    def _load_registry(self) -> dict:
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                return json.load(f)
        return {}

    def _save_registry(self, registry: dict):
        with open(self._registry_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)

    async def save_document(
        self,
        filename:     str,
        contents:     bytes,
        content_type: str,
        size_bytes:   int,
    ) -> tuple[DocumentMetadata, str, list[dict]]:
        """
        Full pipeline: save → parse → extract → index → graph → alert.
        Returns (DocumentMetadata, financial_summary, contradictions).
        """
        meta = DocumentMetadata(
            filename     = filename,
            size_bytes   = size_bytes,
            content_type = content_type,
        )

        # 1. Save raw file
        file_path = self.upload_dir / f"{meta.doc_id}_{filename}"
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(contents)

        # 2. Parse with financial parser
        parsed = self.parser.parse(file_path, filename)

        # 3. Extract financial metrics
        metrics           = self.extractor.extract(parsed.raw_text, parsed.doc_type)
        financial_summary = self.extractor.format_for_display(metrics)

        # 4. Enrich metadata
        meta.doc_type              = parsed.doc_type
        meta.ticker                = parsed.ticker
        meta.fiscal_period         = parsed.fiscal_period
        meta.metrics_found         = parsed.metrics_found
        meta.has_tables            = len(parsed.tables) > 0
        meta.extraction_confidence = metrics.confidence

        # 5. Index into FAISS
        chunk_count      = self._index_parsed_document(meta.doc_id, parsed)
        meta.chunk_count = chunk_count

        # 6. Add to Neo4j graph + auto-detect relationships
        self.graph.add_document(
            doc_id        = meta.doc_id,
            filename      = filename,
            doc_type      = meta.doc_type or "general",
            ticker        = meta.ticker,
            fiscal_period = meta.fiscal_period,
            metrics_found = meta.metrics_found or [],
            has_tables    = meta.has_tables,
            chunk_count   = chunk_count,
        )

        # 7. Contradiction check
        contradictions = []
        if meta.ticker and meta.ticker != "unknown":
            contradictions = self.contradiction.check_on_upload(
                new_doc_id   = meta.doc_id,
                new_doc_text = parsed.raw_text,
                ticker       = meta.ticker,
            )

        # 8. Queue proactive alert checks (async background worker)
        queue_document_analysis(
            doc_id        = meta.doc_id,
            filename      = filename,
            ticker        = meta.ticker or "UNKNOWN",
            doc_type      = meta.doc_type or "general",
            fiscal_period = meta.fiscal_period or "unknown",
            metrics_found = meta.metrics_found or [],
            raw_text      = parsed.raw_text,
        )

        # 9. Register metadata
        registry              = self._load_registry()
        registry[meta.doc_id] = meta.model_dump()
        self._save_registry(registry)

        return meta, financial_summary, contradictions

    async def list_documents(self) -> list[DocumentMetadata]:
        registry = self._load_registry()
        return [DocumentMetadata(**v) for v in registry.values()]

    async def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        registry = self._load_registry()
        if doc_id not in registry:
            return None
        return DocumentMetadata(**registry[doc_id])

    async def delete_document(self, doc_id: str) -> bool:
        registry = self._load_registry()
        if doc_id not in registry:
            return False

        meta      = registry[doc_id]
        file_path = self.upload_dir / f"{doc_id}_{meta['filename']}"
        if file_path.exists():
            file_path.unlink()

        self.rag.delete_index(doc_id)
        self.graph.delete_document(doc_id)
        del registry[doc_id]
        self._save_registry(registry)
        return True

    def _index_parsed_document(self, doc_id: str, parsed) -> int:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings

        if not parsed.chunks:
            return 0

        embeddings = HuggingFaceEmbeddings(
            model_name   = "all-MiniLM-L6-v2",
            model_kwargs = {"device": "cpu"},
        )

        for chunk in parsed.chunks:
            chunk.metadata["doc_id"] = doc_id

        index_path  = Path(settings.FAISS_INDEX_DIR) / doc_id
        vectorstore = FAISS.from_documents(parsed.chunks, embeddings)
        vectorstore.save_local(str(index_path))

        return len(parsed.chunks)