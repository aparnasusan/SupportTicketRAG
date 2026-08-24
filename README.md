# Support Ticket RAG Assistant

This project begins with semantic search and now includes a Phase 2 local RAG prototype. Given a new support issue, it retrieves similar historical tickets and asks a local LLM to produce an evidence-grounded suggested resolution.

## What each file does

| File | Purpose |
| --- | --- |
| `data/tickets.csv` | Thirty synthetic historical tickets used as the retrieval corpus. |
| `ingest.py` | Reads the CSV, turns each row into an embedding-ready document, and stores the vectors in ChromaDB. |
| `search.py` | Embeds a new issue and retrieves the nearest historical tickets. |
| `rag.py` | Retrieves the nearest tickets and sends them as evidence to a local Ollama model. |
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

## Phase 2: local RAG with Ollama

Phase 2 preserves the tested semantic retrieval path. `rag.py` first retrieves the top three tickets, then provides only those tickets and the user issue to a local LLM. The model is instructed to cite the ticket IDs it used and to state when evidence is insufficient.

```text
User issue → semantic retrieval → top 3 tickets → Ollama → grounded response
```

### Install Ollama

Install Ollama for Windows, then download the small local model used by default:

```bash
ollama pull llama3.2:3b
```

No API key or cloud account is needed. The model download requires several gigabytes of local disk space.

### Run a grounded resolution

First create the ticket index if it does not already exist:

```bash
python ingest.py
```

Then run:

```bash
python rag.py "Customer cannot log in after resetting their password"
```

To use another installed Ollama model for an experiment:

```bash
python rag.py "Customer cannot log in after resetting their password" --model <model-name>
```

The app prints the generated response and the retrieved ticket IDs with similarity scores. This lets you inspect whether the answer is supported by its evidence.

### Manual RAG behavior checks

These checks validate both the retrieval path and the grounding behavior of the
generated response. They are qualitative checks, not yet a formal evaluation set.

| Scenario | Query | Expected behavior | Result |
| --- | --- | --- | --- |
| Direct evidence | `Customer cannot log in after resetting their password` | Identify the stale password-reset session, recommend the SR001 resolution, and cite only `SR001`. | Passed |
| Unsupported question | `How do I change my organization's logo?` | State that the retrieved evidence is insufficient and do not invent a resolution or ticket citation. | Passed |

### Grounding improvement

The first password-reset response retrieved the correct ticket but cited other,
merely related authentication tickets and described the direct evidence as uncertain.
The prompt was then strengthened to distinguish direct matches from topical matches,
require only directly supporting citations, and prevent evidence limitations that
contradict the retrieved ticket. The corrected response cited `SR001` alone and
reported no material evidence limitation.

This before-and-after test is an example of **faithfulness**: a RAG response should
accurately reflect the evidence that was retrieved, not merely produce a plausible
answer.

## Next phases (not implemented yet)

1. Add a lightweight Streamlit interface.
2. Compare retrieval strategies with a labeled evaluation set.
3. Add a hosted-provider implementation for a production deployment comparison.


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
