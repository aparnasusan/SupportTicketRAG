# Interview Notes: Retrieval Evaluation Story

## Short version

I built a support-ticket RAG assistant with local semantic retrieval, a ChromaDB
vector store, and a local Ollama model. Rather than judging quality by whether one
answer looked plausible, I created a small labeled retrieval evaluation set with
supported and unsupported queries. I measured Recall@3, Mean Reciprocal Rank (MRR),
and whether unsupported queries were safely blocked before generation.

The baseline achieved 100% Recall@3 across 11 supported cases, but Data Sync had a
lower MRR because one expected ticket was not ranked first. I hypothesized that
embedding resolution text together with a customer's symptom could add ranking
noise. I changed the embedding input to use product and issue text only, retained
the resolution as metadata for generation, and reran the same evaluation set.

That improved overall MRR from 0.939 to 0.955 and Data Sync MRR from 0.778 to 0.833,
while retaining 100% Recall@3 and 2/2 correct unsupported-query abstentions. I did
not add a reranker immediately because the evaluation set is still small and
synthetic; optimizing around a single remaining case would risk overfitting. The
next step would be to expand the labeled dataset, then compare metadata filters or
a reranker using the same metrics.

## Problem → action → result → judgment

| Step | What to say |
| --- | --- |
| Problem | “A top-three retrieval check was not enough: I needed to know whether expected tickets ranked highly and whether irrelevant queries were stopped safely.” |
| Action | “I created a labeled evaluation set and an evaluation script that reused the production retrieval function.” |
| Measures | “I tracked Recall@3, MRR, category-level MRR, and unsupported-query abstentions.” |
| Finding | “All supported cases were retrievable in the top three, but Data Sync ranking was weaker.” |
| Experiment | “I removed resolution text from the embedding input, while preserving it as metadata for the RAG response.” |
| Result | “Overall MRR improved from 0.939 to 0.955; Data Sync MRR improved from 0.778 to 0.833; Recall@3 stayed at 100%.” |
| Judgment | “I deferred a reranker until I have more realistic labeled data, to avoid overfitting one example.” |

## Likely follow-up questions

### Why did you keep the resolution out of the embedding text?

The incoming query describes a customer symptom. Resolution text contains internal
remediation language that may be semantically close to an unrelated ticket and
distort symptom matching. Once a ticket is retrieved, its resolution is still
available as evidence for the LLM.

### Is the score a confidence probability?

No. It is a retrieval ranking score, not a probability that the resolution is
correct. The threshold is an empirical guardrail that must be revisited as the data
and embedding model change.

### Why not add a reranker now?

A reranker adds latency, cost or local-compute requirements, and operational
complexity. With a small synthetic test set, a large change could optimize one
example rather than generalize. A larger labeled dataset is the right prerequisite.

### What would you do for production?

Use a larger, privacy-reviewed ticket corpus; expand the labeled evaluation set;
evaluate retrieval by product and incident type; add observability and feedback;
then compare filters, hybrid retrieval, and reranking under measured latency and
quality targets.
