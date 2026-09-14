"""Search the persistent support-ticket semantic index."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import sqlite3
from chromadb.errors import ChromaError
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import EMBEDDING_MODEL, Settings, get_settings
from dependencies import open_ticket_collection
from errors import ServiceError, RetrievalUnavailable
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


def retrieve_tickets(issue: str, top_k: int = 3, *, settings: Settings | None = None) -> list[RetrievedTicket]:
    """Return the closest historical tickets without deciding how to display them."""
    collection = open_ticket_collection(settings or get_settings())

    try:
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=HuggingFaceEmbedding(model_name=EMBEDDING_MODEL),
        )
        results = index.as_retriever(similarity_top_k=top_k).retrieve(issue)
    except (ChromaError, OSError, sqlite3.Error) as error:
        raise RetrievalUnavailable() from error
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
    try:
        search(args.issue, top_k=args.top_k)
    except ServiceError as error:
        parser.exit(1, f"{error}\n")
