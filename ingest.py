"""Load support tickets from CSV and create a persistent ChromaDB semantic index."""

from __future__ import annotations

import argparse

import chromadb
import pandas as pd
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import PROJECT_DIR, EMBEDDING_MODEL, get_settings

DATA_PATH = PROJECT_DIR / "data" / "tickets.csv"


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
    required_columns = {"ticket_id", "product", "issue", "resolution"}
    missing = required_columns.difference(tickets.columns)
    if missing:
        raise ValueError(f"tickets.csv is missing required columns: {sorted(missing)}")

    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
            print("Removed the existing ticket index.")
        except ValueError:
            pass

    collection = client.get_or_create_collection(settings.chroma_collection)
    if collection.count() > 0:
        print("An index already exists. Run `python ingest.py --reset` to rebuild it.")
        return

    documents = [ticket_to_document(ticket) for _, ticket in tickets.iterrows()]
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
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
    build_index(reset=parser.parse_args().reset)
