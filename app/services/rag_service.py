import os
import pickle
from pathlib import Path
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.schema import Document

from app.core.config import settings


class RAGService:
    """
    Handles the full RAG pipeline:
      - Parse  → extract text from PDF / TXT / DOCX
      - Chunk  → split into overlapping pieces
      - Embed  → convert chunks to vectors (runs locally, no API key needed)
      - Index  → store in FAISS on disk
      - Search → find relevant chunks for a question
    """

    def __init__(self):
        self.index_dir = Path(settings.FAISS_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = Path(settings.UPLOAD_DIR)

        # Runs fully locally using sentence-transformers — no API key needed
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )
        self._store_cache: dict[str, FAISS] = {}

    # ── Index a document on upload ────────────────────────────────

    def index_document(self, doc_id: str, filename: str) -> int:
        """
        Parse → chunk → embed → save FAISS index for one document.
        Returns the number of chunks created.
        """
        file_path = self._find_file(doc_id, filename)
        if not file_path:
            raise FileNotFoundError(f"File not found for doc_id={doc_id}")

        # Step 1: Parse
        raw_docs = self._parse(file_path, filename)

        # Step 2: Chunk
        chunks = self.splitter.split_documents(raw_docs)

        # Attach doc_id metadata to every chunk so we can filter later
        for i, chunk in enumerate(chunks):
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["filename"] = filename
            chunk.metadata["chunk_index"] = i

        # Step 3 + 4: Embed and save FAISS index
        index_path = self.index_dir / doc_id
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        vectorstore.save_local(str(index_path))

        return len(chunks)

    # ── Search across indexed documents ──────────────────────────

    def search(
        self,
        question: str,
        doc_ids: List[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Embed the question, search FAISS indexes, return top_k chunks with scores.
        If doc_ids is None, searches all indexed documents.
        """
        # Clean user-provided ids from API clients (Swagger often sends empty strings).
        cleaned_doc_ids = [d.strip() for d in (doc_ids or []) if d and d.strip()]
        all_indexed = self._all_indexed_doc_ids()

        if cleaned_doc_ids:
            # Keep only doc_ids that have a complete FAISS index on disk.
            targets = [d for d in cleaned_doc_ids if (self.index_dir / d / "index.faiss").exists()]
            # If all provided ids are invalid, fall back to all indexed docs.
            if not targets:
                targets = all_indexed
        else:
            targets = all_indexed

        if not targets:
            return []

        # Search each doc index independently and combine top global matches.
        # This avoids repeatedly merging FAISS stores on every query.
        all_results: List[Tuple[Document, float]] = []
        per_doc_k = max(1, min(top_k, 4))

        for doc_id in targets:
            store = self._get_or_load_store(doc_id)
            if store is None:
                continue
            all_results.extend(store.similarity_search_with_score(question, k=per_doc_k))

        if not all_results:
            return []

        all_results.sort(key=lambda item: item[1])
        return all_results[:top_k]

    # ── Delete an index ───────────────────────────────────────────

    def delete_index(self, doc_id: str):
        """Remove FAISS index when a document is deleted."""
        index_path = self.index_dir / doc_id
        if index_path.exists():
            import shutil
            shutil.rmtree(index_path)
        self._store_cache.pop(doc_id, None)

    # ── Internal helpers ──────────────────────────────────────────

    def _parse(self, file_path: Path, filename: str) -> List[Document]:
        """Load raw text from PDF, TXT, or DOCX."""
        ext = filename.lower().split(".")[-1]
        if ext == "pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == "docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def _find_file(self, doc_id: str, filename: str) -> Path:
        """Locate the uploaded file on disk."""
        path = self.upload_dir / f"{doc_id}_{filename}"
        return path if path.exists() else None

    def _all_indexed_doc_ids(self) -> List[str]:
        """Return all doc_ids that have a FAISS index on disk."""
        if not self.index_dir.exists():
            return []
        return [
            p.name
            for p in self.index_dir.iterdir()
            if p.is_dir() and (p / "index.faiss").exists()
        ]

    def _get_or_load_store(self, doc_id: str) -> FAISS | None:
        if doc_id in self._store_cache:
            return self._store_cache[doc_id]

        index_path = self.index_dir / doc_id
        index_file = index_path / "index.faiss"
        if not index_file.exists():
            return None

        store = FAISS.load_local(
            str(index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self._store_cache[doc_id] = store
        return store