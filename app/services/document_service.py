import json
import aiofiles
from pathlib import Path
from typing import Optional

from app.models.schemas import DocumentMetadata
from app.core.config import settings
from app.services.rag_service import RAGService


class DocumentService:
    """
    Handles file persistence.
    Phase 2: now also builds the FAISS index on every upload.
    """

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.upload_dir / "_registry.json"
        self.rag = RAGService()

    # ── Internal registry helpers ─────────────────────────────────

    def _load_registry(self) -> dict:
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                return json.load(f)
        return {}

    def _save_registry(self, registry: dict):
        with open(self._registry_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)

    # ── Public methods ────────────────────────────────────────────

    async def save_document(
        self,
        filename: str,
        contents: bytes,
        content_type: str,
        size_bytes: int,
    ) -> DocumentMetadata:
        """Write file to disk, index into FAISS, register metadata."""
        meta = DocumentMetadata(
            filename=filename,
            size_bytes=size_bytes,
            content_type=content_type,
        )

        # Save file to disk
        file_path = self.upload_dir / f"{meta.doc_id}_{filename}"
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(contents)

        # ── NEW in Phase 2: index into FAISS ──────────────────────
        chunk_count = self.rag.index_document(meta.doc_id, filename)
        meta.chunk_count = chunk_count
        # ─────────────────────────────────────────────────────────

        # Register metadata
        registry = self._load_registry()
        registry[meta.doc_id] = meta.model_dump()
        self._save_registry(registry)

        return meta

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

        meta = registry[doc_id]
        file_path = self.upload_dir / f"{doc_id}_{meta['filename']}"
        if file_path.exists():
            file_path.unlink()

        # ── NEW in Phase 2: remove FAISS index too ────────────────
        self.rag.delete_index(doc_id)
        # ─────────────────────────────────────────────────────────

        del registry[doc_id]
        self._save_registry(registry)
        return True