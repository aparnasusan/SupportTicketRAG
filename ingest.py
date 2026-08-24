"""Load support tickets from CSV and create a persistent ChromaDB semantic index."""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
import pandas as pd
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "tickets.csv"
CHROMA_PATH = PROJECT_DIR / "storage" / "chroma"
COLLECTION_NAME = "support_tickets"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def ticket_to_document(ticket: pd.Series) -> Document:
    """Represent one ticket as readable retrieval text plus filterable metadata."""
    # Bracket access is deliberate: `Series.product` is a Pandas method, not
    # the CSV value in the "product" column.
    ticket_id = ticket["ticket_id"]
    product = ticket["product"]
    issue = ticket["issue"]
    resolution = ticket["resolution"]
    text = (
        f"Product: {product}\n"
        f"Issue: {issue}\n"
        f"Resolution: {resolution}"
    )
    return Document(
        text=text,
        metadata={
            "ticket_id": ticket_id,
            "product": product,
            "issue": issue,
            "resolution": resolution,
        },
        id_=ticket_id,
    )


def build_index(reset: bool) -> None:
    tickets = pd.read_csv(DATA_PATH)
    required_columns = {"ticket_id", "product", "issue", "resolution"}
    missing = required_columns.difference(tickets.columns)
    if missing:
        raise ValueError(f"tickets.csv is missing required columns: {sorted(missing)}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Removed the existing ticket index.")
        except ValueError:
            pass

    collection = client.get_or_create_collection(COLLECTION_NAME)
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
    print(f"Indexed {len(documents)} tickets in {CHROMA_PATH}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the support-ticket semantic index.")
    parser.add_argument("--reset", action="store_true", help="Replace the existing index.")
    build_index(reset=parser.parse_args().reset)
