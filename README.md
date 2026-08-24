# Support Ticket RAG Assistant — Phase 1

This first phase is a semantic-search foundation for a future support-resolution RAG application. It does **not** use an LLM. Given a new support issue, it returns the three most similar historical tickets.

## What each file does

| File | Purpose |
| --- | --- |
| `data/tickets.csv` | Thirty synthetic historical tickets used as the retrieval corpus. |
| `ingest.py` | Reads the CSV, turns each row into an embedding-ready document, and stores the vectors in ChromaDB. |
| `search.py` | Embeds a new issue and retrieves the nearest historical tickets. |
| `requirements.txt` | Python dependencies for the retrieval prototype. |

## Retrieval flow

```text
tickets.csv → document text → embedding model → ChromaDB
new issue → same embedding model → similarity search → top 3 tickets
```

Each ticket is represented for embedding as:

```text
Product: <product>
Issue: <issue>
Resolution: <resolution>
```

This keeps related symptoms and solutions together in the vector representation. The original fields are also stored as metadata, so search results are readable now and can later be filtered by product.

## Setup

Use Python 3.10 or newer. From this project directory:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Windows with Python 3.12, use ChromaDB 1.x as listed in `requirements.txt`.
Older ChromaDB 0.6 releases can attempt a local C++ build and fail when Microsoft
C++ Build Tools are not installed.

The first run downloads the open-source embedding model (`BAAI/bge-small-en-v1.5`).

## Run the prototype

Create the persistent index:

```bash
python ingest.py
```

Search for similar tickets:

```bash
python search.py "Customer cannot log in after resetting their password"
```

To rebuild the index after editing the CSV:

```bash
python ingest.py --reset
```

## Why this architecture?

- **Embeddings** map the meaning of ticket text to vectors, so a query does not need to share exact keywords with a historical ticket.
- **ChromaDB** provides a simple persistent local vector store for learning and iteration.
- **LlamaIndex** keeps ingestion and retrieval code modular, while leaving room for a later RAG layer.
- **No LLM in Phase 1** makes retrieval quality easy to inspect and evaluate before generated answers add another source of error.

## Next phases (not implemented yet)

1. Use retrieved tickets as evidence for a grounded LLM response.
2. Add a lightweight Streamlit interface.
3. Compare retrieval strategies with a labeled evaluation set.


## Manual retrieval checks

These queries were used to confirm that semantic retrieval returns the intended
historical incident as the top result.

| Query | Expected top ticket | Result |
| --- | --- | --- |
| `Customer cannot log in after resetting their password` | `SR001` | Passed |
| `My credit card payment was declined but the card is valid` | `SR007` | Passed |
| `The application is showing a service unavailable error` | `SR025` | Passed |

These checks are qualitative validation only. A later evaluation phase will use
a larger labeled query set and metrics such as Recall@K and MRR.