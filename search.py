"""Search the persistent support-ticket semantic index."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_PATH = PROJECT_DIR / "storage" / "chroma"
COLLECTION_NAME = "support_tickets"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# This is an initial abstention threshold, calibrated from the Phase 1/2 manual
# checks. Revisit it after running a larger labeled evaluation set.
MIN_RETRIEVAL_SCORE = 0.50


@dataclass(frozen=True)
class RetrievedTicket:
    """A historical ticket returned by semantic retrieval."""

    ticket_id: str
    product: str
    issue: str
    resolution: str
    similarity_score: float


def retrieve_tickets(issue: str, top_k: int = 3) -> list[RetrievedTicket]:
    """Return the closest historical tickets without deciding how to display them."""
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
    return [
        RetrievedTicket(
            ticket_id=str(result.node.metadata["ticket_id"]),
            product=str(result.node.metadata["product"]),
            issue=str(result.node.metadata["issue"]),
            resolution=str(result.node.metadata["resolution"]),
            similarity_score=float(result.score or 0.0),
        )
        for result in results
    ]


def has_sufficient_evidence(tickets: list[RetrievedTicket]) -> bool:
    """Return whether the strongest retrieved record clears the abstention gate.

    The score is a retrieval ranking score, not a probability that a proposed
    resolution is correct. This gate simply prevents generation when every
    retrieved record is too weak to treat as useful evidence.
    """
    return bool(tickets) and tickets[0].similarity_score >= MIN_RETRIEVAL_SCORE


def search(issue: str, top_k: int = 3) -> None:
    """Print a readable command-line view of semantic retrieval results."""
    results = retrieve_tickets(issue, top_k=top_k)

    print(f"\nQuery: {issue}\n")
    print(f"Top {len(results)} similar historical tickets:\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result.ticket_id} | {result.product} | similarity: {result.similarity_score:.3f}")
        print(f"   Issue: {result.issue}")
        print(f"   Resolution: {result.resolution}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search historical support tickets semantically.")
    parser.add_argument("issue", help="Natural-language description of the support issue.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of matching tickets to return.")
    args = parser.parse_args()
    search(args.issue, top_k=args.top_k)
