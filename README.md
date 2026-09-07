# Support Ticket RAG Assistant

This project begins with semantic search and now includes a Phase 2 local RAG prototype. Given a new support issue, it retrieves similar historical tickets and asks a local LLM to produce an evidence-grounded suggested resolution.

## What each file does

| File | Purpose |
| --- | --- |
| `data/tickets.csv` | Thirty synthetic historical tickets used as the retrieval corpus. |
| `ingest.py` | Reads the CSV, turns each row into an embedding-ready document, and stores the vectors in ChromaDB. |
| `search.py` | Embeds a new issue and retrieves the nearest historical tickets. |
| `rag.py` | Retrieves the nearest tickets and sends them as evidence to a local Ollama model. |
| `app.py` | Provides a lightweight Streamlit interface over the tested RAG pipeline. |
| `evaluate.py` | Runs labeled retrieval checks and reports Recall@K, MRR, and unsupported-query abstention behavior. |
| `data/evaluation_cases.json` | Version-controlled queries with their expected ticket IDs or an expected abstention. |
| `docs/INTERVIEW_NOTES.md` | Concise evidence-based talking points and likely interview follow-up answers. |
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
```

The embedding represents the support symptom a customer would describe. The ticket
resolution is retained as metadata and supplied to the LLM only after a ticket is
retrieved. This keeps implementation-specific resolution text from distorting issue
matching, while preserving it for the evidence-grounded response. The original fields
are also stored as metadata, so search results are readable now and can later be
filtered by product.

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

## Phase 3: Streamlit interface

Phase 3 adds a small web interface without duplicating the AI logic. `app.py` calls
the same `generate_resolution()` function used by the command-line application.
The interface lets a user enter an issue, choose the number of retrieved tickets,
read the generated response, and inspect the underlying ticket evidence.

Run the app from the project folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit opens the local application in a browser. Ollama must be installed and the
configured local model must be downloaded before submitting an issue.

The project pins Streamlit below version 1.53 because later releases require newer
Starlette internals than the local ChromaDB dependency stack provides.

## Next phases (not implemented yet)

1. Compare retrieval strategies with the labeled evaluation set.
2. Add metadata filtering and/or reranking if the metrics identify weak categories.
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

## Phase 4: evaluation and retrieval safeguards

Phase 4 adds a small, version-controlled evaluation set and an abstention gate.
This separates two questions that are easy to conflate in a RAG system:

- **Retrieval quality:** did the expected historical ticket appear in the top results?
- **Generation safety:** was the retrieved evidence relevant enough to ask the LLM for a resolution?

Run the evaluation after building the index:

```bash
python evaluate.py
```

The script uses the same `retrieve_tickets()` function as the command-line tool and
Streamlit interface. It reports:

- **Recall@3**, the share of supported test queries whose expected ticket occurs in
  the three retrieved results.
- **MRR (Mean Reciprocal Rank)**, which rewards placing the expected ticket near the
  top of the ranking.
- **Unsupported abstentions**, where the retrieval gate should prevent generation.
- Per-category MRR, to reveal weaker ticket domains.

`rag.py` now checks `has_sufficient_evidence()` before it calls Ollama. The current
threshold is `0.50`, selected after the first labeled evaluation. In that run, the
lowest supported-case top score was `0.562`, while the highest unsupported-case top
score was `0.480`; `0.50` therefore separated those initial examples. It is a
retrieval ranking threshold, not a probability that a resolution is correct. Treat
it as an initial baseline: expand the evaluation set, inspect failures, then adjust
and document the threshold if the evidence supports doing so.

### Baseline evaluation results

After tuning the threshold, the first evaluation run produced:

| Metric | Result | Interpretation |
| --- | --- | --- |
| Recall@3 | 100% (11/11) | Each supported query returned its expected ticket in the first three results. |
| MRR | 0.939 | Most expected tickets ranked first; one Data Sync case ranked lower. |
| Unsupported abstentions | 2/2 | Both unsupported queries were blocked before the local LLM was called. |

The Data Sync category has an MRR of `0.778`, whereas the other tested categories
are `1.000`. The next retrieval experiment should focus on why Data Sync tickets
are semantically close to one another—for example, by testing more labels, metadata
filters, or a reranker—rather than changing the threshold again without evidence.

### Retrieval experiment: symptom-focused embeddings

The initial index embedded each ticket's product, issue, and resolution together.
The first measured experiment instead embeds only product and issue, because customer
queries describe symptoms rather than internal remediation steps. Resolutions remain
stored as metadata and are still provided to the LLM after retrieval. Rebuild the
index and rerun `python evaluate.py` to compare this change against the baseline
above; do not update the baseline metrics unless the new run is recorded.

The experiment improved the recorded metrics without changing the evaluation cases:

| Metric | Baseline | Symptom-focused embedding | Change |
| --- | ---: | ---: | ---: |
| Recall@3 | 100% (11/11) | 100% (11/11) | No loss in coverage |
| MRR | 0.939 | 0.955 | Improved ranking |
| Data Sync MRR | 0.778 | 0.833 | Improved ranking |
| Unsupported abstentions | 2/2 | 2/2 | No loss in safe behavior |

For E006, the expected ticket `SR017` moved from third to second place; it did not
yet become the top result. This is a meaningful improvement, but not a reason to
claim the issue is solved. The next evidence-based experiment would be a reranker or
metadata-aware retrieval, evaluated against a larger Data Sync set.
