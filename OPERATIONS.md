# Local operations and Phase 6 decisions

Phase 6 adds operational safeguards to the local support assistant. It does not
introduce public hosting, authentication, a reranker, or paid inference.
The subsequent local container setup is covered in [DOCKER.md](DOCKER.md).
The API, CLI, and Streamlit still use the same retrieval and generation functions.

## Configuration

Copy `.env.example` to `.env` only if you need overrides. The optional file lives
in the project directory. Process environment variables override that file, which
overrides defaults. Settings are validated and cached; restart the API or Streamlit
process after changing them. Tests can pass a validated Settings object into the
application factory without changing global environment values.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHROMA_PATH` | `storage/chroma` | Relative paths resolve from the project directory, not the terminal's working directory. |
| `CHROMA_COLLECTION` | `support_tickets` | Existing collection used by retrieval and readiness; ingestion uses the same setting. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | HTTP(S) origin only; no credentials, query, or API path. |
| `OLLAMA_MODEL` | `llama3.2:3b` | Server-selected installed model. API callers cannot override it. |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Positive, finite network timeout for each chat call. |
| `HEALTHCHECK_TIMEOUT_SECONDS` | `3` | Positive, finite network timeout for the Ollama probe. |
| `CORS_ALLOWED_ORIGINS` | `[]` | JSON array of exact browser origins; wildcards are rejected. |

Example PowerShell override:

```powershell
$env:CORS_ALLOWED_ORIGINS = '["http://localhost:3000"]'
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
```

The application writes its own structured completion logs. Disabling Uvicorn's
separate access log avoids duplicate logs and raw URL/query-string logging.
Send support text in the JSON request body, never in URLs or request IDs.

Invalid settings fail startup with field names or a safe format error, without
echoing configuration values. Missing dependencies do not prevent the API from
starting; readiness reports their condition.

The embedding model, retrieval threshold, evidence-selection thresholds, prompts,
temperature, seed, and one-repair policy remain fixed at the evaluated settings.
Changing the Ollama model is still an explicit local experiment requiring evaluation.
The CLI model option and Streamlit model picker remain available for that purpose.

**Interview point:** Configuration describes where services live and how to connect
to them. Evaluated RAG policy should not drift through casual environment overrides.

## Start and verify locally

From the repository directory, with the virtual environment activated:

```powershell
python -m pip install -r requirements.txt
python ingest.py
ollama pull llama3.2:3b
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Ollama must be running. Model downloads are explicit setup steps; health checks and
API startup do not download models or build an index. Initial embedding setup may
download the embedding model. Keep the existing model cache for offline use.

In a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/health/ready
$body = @{ issue = "Customer cannot log in after resetting their password"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/resolutions -Method Post -ContentType "application/json" -Body $body
```

The interactive API reference is available at `http://127.0.0.1:8000/docs`.
Use one process for this local setup. Concurrency, latency optimization, and a public
deployment process strategy are future work.

## Liveness and readiness

| Endpoint | What it checks | Success | Failure |
| --- | --- | --- | --- |
| `GET /health` | The API can respond | 200, `{"status":"ok"}` | An unresponsive process cannot serve the check. |
| `GET /health/ready` | Existing Chroma collection is accessible and nonempty; Ollama lists the configured model | 200, `status: ready` | 503, `status: not_ready`, safe component reasons |

The readiness response includes independent `chroma` and `ollama` components.
Both are checked even if one fails. The shared URL prefix groups related routes;
neither endpoint calls the other.

Chroma is embedded persistent storage, not a separate server. The probe checks for
the existing database before opening the client, then gets the configured collection
and counts records. It never creates a collection or ingests documents. Opening an
existing Chroma client uses Chroma's normal database initialization; it is not a
strictly read-only SQLite connection.

The Ollama probe uses `GET /api/tags`. A model being installed does not prove it
can load into memory or generate successfully. Probes do not embed queries, perform
vector search, call chat, or verify semantic answer quality.

Checks run in a synchronous endpoint, so FastAPI places blocking work in its thread
pool. The configured network timeout does not impose a hard deadline on local
Chroma disk I/O or guarantee a total request deadline. Chat's timeout also applies
per call; the existing repair attempt can require a second call.

Readiness is a snapshot, not a guarantee. The resolution endpoint handles actual
failures independently and does not run a readiness check before every request.
An unsupported query can still abstain successfully when Ollama is unavailable,
although overall readiness reports the full service as unavailable.

**Interview point:** A dependency outage should make the service unready without
suggesting that restarting a healthy API process will fix the dependency.

## Browser access

CORS starts with an empty allowlist. Configure only real browser clients.
Origins include scheme, hostname, and port; `localhost` and `127.0.0.1` differ.
The middleware permits GET and POST, handles preflight requests, permits the
content-type and request-ID headers, and exposes the response request ID.
Credentials remain disabled.

Streamlit calls the shared Python function directly and needs no API CORS origin.
Same-origin API documentation works without an allowlist entry.
CORS controls browser access to responses. It is not authentication or protection
against non-browser clients.

**Interview point:** Browser permissions are explicit and minimal; they do not
substitute for authentication before future public access.

## Errors and logs

| Situation | HTTP response |
| --- | --- |
| Grounded answer | 200 with answer and retrieved evidence |
| Weak evidence | 200 with the existing abstention and evidence |
| Invalid input | 422 with safe locations/types and generic validation messages |
| Missing/empty index or known retrieval/Ollama failure | 503 with a generic service message |
| Invalid model response, including a second grounding failure | 503 |
| Unexpected application defect | 500 with a generic message |

The successful resolution schema is unchanged. Issue length is checked after
trimming. Validation responses omit input values, validator context, and arbitrary
unknown field names that might contain support content.

Shared code raises application exceptions rather than terminating the process.
The API translates them to HTTP responses. CLI commands print safe errors and exit
unsuccessfully; Streamlit displays safe service errors.

Logs contain UTC timestamps, request IDs, method, route template, status, duration,
and safe failure categories. They omit request bodies, upstream error bodies,
raw exception text, and arbitrary URL paths/query strings. Incoming request IDs
must contain 1-64 ASCII letters, digits, dots, underscores, or hyphens; otherwise a
new ID is generated. Callers should use opaque correlation IDs, not support text.

Middleware order is intentional: request tracking surrounds CORS, which surrounds
the unexpected-error boundary. Thus preflights get IDs/logs and allowed origins
receive CORS headers on endpoint errors as well as successful responses.

**Tradeoff:** Safe categories give less detail than arbitrary exception dumps.
Reproduce defects locally with synthetic inputs rather than enabling body logging.
Other libraries or hosting tools may have their own logs; this policy describes
the dedicated API logger.

**Interview point:** Operational failures, invalid input, and software defects have
different outcomes, while none requires exposing customer content.

## Troubleshooting

| Safe reason | Next check |
| --- | --- |
| `index_unavailable` | Verify path/collection settings; run ingestion for the intended index. |
| `retrieval_unavailable` | Verify storage access, index compatibility, and embedding-model availability. |
| `ollama_unreachable` / `ollama_unavailable` | Check the Ollama process, base URL, and network reachability. |
| `model_missing` | Install the configured model explicitly. |
| `invalid_model_list` | Confirm the URL points to Ollama and its model-list API responds correctly. |
| `invalid_model_response` | Retry manually, inspect with synthetic evidence, and evaluate any model change. |
| `probe_failed` / `internal_error` | Correlate by request ID and reproduce locally; these require investigation. |

The index persists under the configured Chroma directory. Preserve the CSV and
configuration so the synthetic index can be rebuilt. Rebuilds are explicit:
`python ingest.py --reset` replaces the configured collection. Do not rebuild
during API startup or while serving traffic.

## Validation and limits

```powershell
python -m unittest discover -v
python evaluate.py
python -m pip check
```

The tests mock dependency I/O and generation. They cover configuration validation,
readiness failure modes, CORS, privacy, request IDs, successful and error contracts,
abstention without inference, and the one-repair boundary.

The 13-case evaluation measures retrieval and the relevance gate, not end-to-end
semantic correctness. The recorded symptom-focused baseline is Recall@3 100%
(11/11), MRR 0.955, Data Sync MRR 0.833, and unsupported abstentions 2/2.
Compare per-case results as well as aggregate metrics after retrieval changes.

For a real inference smoke test, check a direct password-reset issue and an
unsupported organization-logo question. Confirm that the direct answer is grounded
in SR001 and that the unsupported answer abstains without calling the model.
A deterministic seed improves repeatability; it is not a universal guarantee of
identical text across hardware or model versions.

This remains a local portfolio project using synthetic tickets. Public deployment,
authentication, rate limiting, broader evaluation, and measured performance
optimization remain separate milestones.
