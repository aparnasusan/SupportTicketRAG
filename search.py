"""Search the persistent support-ticket semantic index."""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_PATH = PROJECT_DIR / "storage" / "chroma"
COLLECTION_NAME = "support_tickets"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def search(issue: str, top_k: int = 3) -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except ValueError as error:
        raise SystemExit("No index found. Run `python ingest.py` first.") from error

    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=HuggingFaceEmbedding(model_name=EMBEDDING_MODEL),
    )
    results = index.as_retriever(similarity_top_k=top_k).retrieve(issue)

    print(f"\nQuery: {issue}\n")
    print(f"Top {len(results)} similar historical tickets:\n")
    for rank, result in enumerate(results, start=1):
        metadata = result.node.metadata
        print(f"{rank}. {metadata['ticket_id']} | {metadata['product']} | similarity: {result.score:.3f}")
        print(f"   Issue: {metadata['issue']}")
        print(f"   Resolution: {metadata['resolution']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search historical support tickets semantically.")
    parser.add_argument("issue", help="Natural-language description of the support issue.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of matching tickets to return.")
    args = parser.parse_args()
    search(args.issue, top_k=args.top_k)
