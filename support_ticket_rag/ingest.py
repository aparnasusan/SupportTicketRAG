"""Load support tickets from CSV and create a persistent ChromaDB semantic index."""

from __future__ import annotations

import argparse
import re

import chromadb
import pandas as pd
from chromadb.errors import NotFoundError
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from support_ticket_rag.config import EMBEDDING_MODEL, PROJECT_DIR, get_settings

DATA_PATH = PROJECT_DIR / "data" / "tickets.csv"


def validate_tickets(tickets: pd.DataFrame) -> None:
    """Reject invalid data before opening or changing persistent storage."""
    required = {"ticket_id", "product", "issue", "resolution"}
    missing = required.difference(tickets.columns)
    if missing:
        raise ValueError(f"tickets.csv is missing required columns: {sorted(missing)}")
    if tickets.empty:
        raise ValueError("tickets.csv must contain at least one ticket.")
    for column in sorted(required):
        if (
            not tickets[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"tickets.csv contains blank or non-text values in {column}.")
    if (
        not tickets["ticket_id"]
        .map(lambda value: re.fullmatch(r"SR[0-9]{3}", value) is not None)
        .all()
    ):
        raise ValueError("Ticket IDs must use the SR001 format.")
    if tickets["ticket_id"].duplicated().any():
        raise ValueError("Ticket IDs must be unique.")


def build_retrieval_text(product: str, issue: str) -> str:
    """Build the text used to match a new issue to a historical ticket.

    Resolutions are deliberately omitted from the embedding input. A customer
    describes a symptom, so matching against historical symptoms avoids a
    resolution's implementation-specific language distorting the ranking.
    The resolution is still retained as metadata for the RAG response.
    """
    return f"Product: {product}\nIssue: {issue}"


def ticket_to_document(ticket: pd.Series) -> Document:
    """Represent one ticket as retrieval text plus filterable metadata."""
    # Bracket access is deliberate: `Series.product` is a Pandas method, not
    # the CSV value in the "product" column.
    ticket_id = ticket["ticket_id"]
    product = ticket["product"]
    issue = ticket["issue"]
    resolution = ticket["resolution"]
    return Document(
        text=build_retrieval_text(product, issue),
        metadata={
            "ticket_id": ticket_id,
            "product": product,
            "issue": issue,
            "resolution": resolution,
        },
        id_=ticket_id,
    )


def build_index(reset: bool) -> None:
    settings = get_settings()
    tickets = pd.read_csv(DATA_PATH)
    validate_tickets(tickets)

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    if not reset:
        try:
            collection = client.get_collection(settings.chroma_collection)
        except NotFoundError:
            collection = None
        if collection is not None and collection.count() > 0:
            print(
                "An index already exists. Run `python -m support_ticket_rag.ingest --reset` to rebuild it."
            )
            return

    # Preflight data conversion and model loading before removing the old index.
    # A later embedding/write failure can still require rerunning ingestion.
    documents = [ticket_to_document(ticket) for _, ticket in tickets.iterrows()]
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
            print("Removed the existing ticket index.")
        except NotFoundError:
            pass

    collection = client.get_or_create_collection(settings.chroma_collection)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    print(f"Indexed {len(documents)} tickets in {settings.chroma_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the support-ticket semantic index.")
    parser.add_argument("--reset", action="store_true", help="Replace the existing index.")
    args = parser.parse_args()
    try:
        build_index(reset=args.reset)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
