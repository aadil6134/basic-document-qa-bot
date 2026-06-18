from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Iterable

import chromadb
from docx import Document
from pypdf import PdfReader
from tqdm import tqdm

from src.config import settings
from src.embeddings import GeminiEmbeddingFunction


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_pages(file_path: Path) -> list[dict]:
    pages: list[dict] = []
    reader = PdfReader(str(file_path))

    for index, page in enumerate(reader.pages):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(
                {
                    "text": text,
                    "metadata": {"source": file_path.name, "page": index + 1},
                }
            )

    return pages


def extract_docx_pages(file_path: Path) -> list[dict]:
    document = Document(str(file_path))
    paragraphs = [clean_text(paragraph.text) for paragraph in document.paragraphs]
    text = clean_text("\n".join(paragraph for paragraph in paragraphs if paragraph))

    if not text:
        return []

    return [{"text": text, "metadata": {"source": file_path.name, "page": 1}}]


def extract_txt_pages(file_path: Path) -> list[dict]:
    text = clean_text(file_path.read_text(encoding="utf-8", errors="ignore"))

    if not text:
        return []

    return [{"text": text, "metadata": {"source": file_path.name, "page": 1}}]


def extract_document(file_path: Path) -> list[dict]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_pages(file_path)
    if suffix == ".docx":
        return extract_docx_pages(file_path)
    if suffix == ".txt":
        return extract_txt_pages(file_path)

    return []


def discover_documents(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []

    return sorted(
        file_path
        for file_path in data_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _find_split_point(text: str, start: int, max_end: int) -> int:
    if max_end >= len(text):
        return len(text)

    window = text[start:max_end]
    split_candidates = [window.rfind(separator) for separator in ("\n\n", "\n", ". ", " ")]
    split_at = max(split_candidates)

    if split_at <= 0:
        return max_end

    return start + split_at + 1


def chunk_extracted_pages(
    pages: list[dict],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[dict]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[dict] = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        start = 0

        while start < len(text):
            end = _find_split_point(text, start, min(start + chunk_size, len(text)))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "source": metadata["source"],
                            "page": metadata["page"],
                            "chunk_range": f"{start}-{end}",
                        },
                    }
                )

            if end >= len(text):
                break

            start = max(end - chunk_overlap, start + 1)

    return chunks


def load_and_chunk_documents(data_dir: Path) -> list[dict]:
    all_pages: list[dict] = []
    documents = discover_documents(data_dir)

    for file_path in tqdm(documents, desc="Extracting documents"):
        try:
            all_pages.extend(extract_document(file_path))
        except Exception as exc:
            print(f"Skipping {file_path.name}: {exc}")

    return chunk_extracted_pages(all_pages)


def make_chunk_id(chunk: dict) -> str:
    metadata = chunk["metadata"]
    stable_key = f"{metadata['source']}:{metadata['page']}:{metadata['chunk_range']}:{chunk['text']}"
    return hashlib.sha256(stable_key.encode("utf-8")).hexdigest()


def build_embedding_function() -> GeminiEmbeddingFunction:
    return GeminiEmbeddingFunction(task_type="retrieval_document")


def save_to_vector_db(chunks: Iterable[dict], db_path: Path, reset: bool = False) -> int:
    chunk_list = list(chunks)

    if not chunk_list:
        return 0

    client = chromadb.PersistentClient(path=str(db_path))

    if reset:
        try:
            client.delete_collection(settings.collection_name)
        except ValueError:
            pass

    collection = client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=build_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    ids = [make_chunk_id(chunk) for chunk in chunk_list]
    documents = [chunk["text"] for chunk in chunk_list]
    metadatas = [chunk["metadata"] for chunk in chunk_list]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunk_list)


def ingest(data_dir: Path = settings.data_dir, db_path: Path = settings.db_path, reset: bool = False) -> int:
    chunks = load_and_chunk_documents(data_dir)
    return save_to_vector_db(chunks, db_path, reset=reset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index local documents into ChromaDB.")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--db-path", type=Path, default=settings.db_path)
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the Chroma collection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = ingest(args.data_dir, args.db_path, reset=args.reset)
    print(f"Indexed {count} chunks into {args.db_path}.")


if __name__ == "__main__":
    main()
