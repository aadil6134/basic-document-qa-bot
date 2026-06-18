from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from google import genai

from src.config import require_api_key, settings
from src.embeddings import GeminiEmbeddingFunction


NOT_FOUND_MESSAGE = "I cannot find the answer in the provided documents."


def build_embedding_function() -> GeminiEmbeddingFunction:
    return GeminiEmbeddingFunction(task_type="RETRIEVAL_QUERY")


def get_collection(db_path: Path = settings.db_path):
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(
        name=settings.collection_name,
        embedding_function=build_embedding_function(),
    )


def format_context(results: dict) -> tuple[str, list[str]]:
    context_blocks: list[str] = []
    citations: list[str] = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for document, metadata in zip(documents, metadatas):
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        citation = f"{source}, Page {page}"
        context_blocks.append(f"[Source: {citation}]\n{document}")

        if citation not in citations:
            citations.append(citation)

    return "\n\n---\n\n".join(context_blocks), citations


def build_prompt(question: str, context_payload: str) -> str:
    return f"""You are a precise document Q&A assistant.
Use ONLY the provided context to answer the user's question.
If the answer cannot be found in the context, say exactly: "{NOT_FOUND_MESSAGE}"
Do not use outside knowledge or make assumptions.
Cite source filenames and page numbers inline next to the facts you use.

CONTEXT:
{context_payload}

QUESTION:
{question}

ANSWER:"""


def query_rag_pipeline(
    user_query: str,
    db_path: Path = settings.db_path,
    k: int = settings.top_k,
) -> dict:
    collection = get_collection(db_path)
    results = collection.query(query_texts=[user_query], n_results=k)
    context_payload, citations = format_context(results)

    if not context_payload.strip():
        return {"answer": NOT_FOUND_MESSAGE, "citations": [], "raw_context": []}

    client = genai.Client(api_key=require_api_key())
    response = client.models.generate_content(
        model=settings.generation_model,
        contents=build_prompt(user_query, context_payload),
    )

    return {
        "answer": response.text.strip(),
        "citations": citations,
        "raw_context": results.get("documents", [[]])[0],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question against the indexed documents.")
    parser.add_argument("question", help="Question to answer from the document collection.")
    parser.add_argument("--db-path", type=Path, default=settings.db_path)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = query_rag_pipeline(args.question, args.db_path, args.top_k)
    print("\nAnswer:\n")
    print(result["answer"])

    if result["citations"]:
        print("\nCitations:")
        for citation in result["citations"]:
            print(f"- {citation}")


if __name__ == "__main__":
    main()
