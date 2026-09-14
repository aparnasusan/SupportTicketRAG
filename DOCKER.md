# Local Docker deployment

The API runs in a Linux container; Ollama stays on the Windows host. Chroma is
embedded in the API process and persists in a named volume. A separate volume
stores the embedding-model download. Streamlit can still run locally.

This topology keeps Ollama's existing model downloads and host hardware support
without adding GPU passthrough or a second inference container.

## Requirements

- Docker Desktop running **Linux containers**.
- Ollama running with the configured model installed.
- Internet access for the initial image build and embedding-model download.
- Disk space for the image, Python/CPU PyTorch dependencies, and model cache.

The Python runtime is 3.12. The image uses CPU PyTorch for embeddings.
Direct dependencies and the core model stack are constrained to the Phase 6
versions. This is not a complete transitive lock; the installed package inventory
is saved inside the image at `/app/installed-requirements.txt`.

## Build, initialize, and start

Run these from the repository directory:

```powershell
docker compose build
docker compose run --rm --no-deps api python -m unittest discover -v
docker compose run --rm --no-deps api python ingest.py
docker compose run --rm --no-deps api python evaluate.py
docker compose up -d
docker compose ps
```

Ingestion is an explicit one-off command and downloads the embedding model on its
first run. It uses the same image, configuration, user, and volumes as the API.
It leaves an existing nonempty index intact. Do not copy the host's Windows
Chroma directory into the image or reset your existing host index.

The API starts even if dependencies are unavailable; readiness remains unsuccessful
until the index is populated and the configured Ollama model is reachable.
Startup does not ingest tickets, pull models, or reset data.

## Verify host Ollama connectivity

Inside a container, `localhost` means that container. Compose therefore defaults
to `http://host.docker.internal:11434`, Docker Desktop's host address.
It intentionally ignores the local `.env` value of `OLLAMA_BASE_URL`, which may
point to host localhost. Override `DOCKER_OLLAMA_BASE_URL` if your topology differs.

```powershell
docker compose run --rm --no-deps api python -c "from dependencies import check_readiness; from config import get_settings; print(check_readiness(get_settings()))"
```

An unreachable Ollama result means that installing Ollama or resolving the host
name alone is insufficient. Confirm Ollama is running and its listening interface
accepts connections from Docker Desktop. Ollama's documented binding control is
`OLLAMA_HOST`. Changing it requires restarting Ollama. A broad binding can expose
its unauthenticated API to other machines; restrict firewall access to the local
Docker path. Do not disable the firewall or publish Ollama to the internet.
No host listener or firewall settings are changed by these project files.

Model presence is checked through Ollama's model-list API. Readiness does not
prove the model can fit into memory or generate a correct answer.

## Smoke test the running API

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/health/ready
$body = @{ issue = "Customer cannot log in after resetting their password"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/v1/resolutions -Method Post -ContentType "application/json" -Body $body
```

Expect the direct resolution to cite SR001 while returning three evidence records.
The organization-logo question should abstain. Compare container retrieval results
against the recorded baseline: Recall@3 100%, MRR 0.955, Data Sync MRR 0.833, and
unsupported abstentions 2/2. Hardware and numerical libraries can affect rankings;
record differences rather than silently adjusting thresholds.

If port 8000 is already used, set `SUPPORT_API_PORT=8001` in the shell or local
`.env`, then recreate the service and use that port in the smoke commands.

## Configuration and persistence

Compose reads interpolation values from the process environment or project
`.env`, but does not copy that file into the image or inject it wholesale.
It explicitly passes model, collection, timeout, and CORS settings.
The container storage path is fixed to its mounted volume. See
[OPERATIONS.md](OPERATIONS.md) for application settings and error semantics.

The API listens on all interfaces **inside** its container, but the published
Windows port is bound to loopback only. CORS stays empty by default.
The process runs as a non-root user, drops Linux capabilities, and prevents
privilege escalation. Named volumes receive writable ownership from the image's
initialized directories when first created.

The Docker health check calls readiness, with a startup grace period for imports.
A failed check marks the container unhealthy; it does not automatically restart it.
For probes that trigger restarts, use liveness rather than dependency readiness.
Health checks do not run inference.

```powershell
docker compose logs --tail 50 api
docker compose stop
docker compose start
docker compose down
```

Stopping, recreating, or removing the container normally preserves named volumes.
**Do not use `docker compose down -v` unless you intend to delete the container's
index and embedding cache.** Host Chroma files are independent of these volumes.

To rebuild only the container index after intentionally updating the synthetic CSV:

```powershell
docker compose stop api
docker compose build
docker compose run --rm --no-deps api python ingest.py --reset
docker compose run --rm --no-deps api python evaluate.py
docker compose up -d
```

The build context is an allowlist. It excludes local notes, Git metadata,
credentials, virtual environments, and host storage. No Docker socket or host
repository is mounted into the running container.

## Interview talking points

- **Image versus data:** The image contains code and dependencies; volumes retain
  the index and model cache across container replacement.
- **Explicit bootstrap:** Index construction is an operator action, avoiding
  destructive or slow startup behavior.
- **Host inference:** Keeping Ollama outside the container reduces local GPU and
  model-management complexity.
- **Operational signals:** Container health checks inspect readiness; liveness
  remains independent of dependency outages.
- **Reproducibility limits:** Version constraints and recorded package inventories
  help reproduce results, but evaluation remains necessary across platforms.
- **Scope:** This is a local deployment, not an authenticated public service.

References: [Docker Desktop host networking](https://docs.docker.com/desktop/features/networking/networking-how-tos/),
[Ollama configuration](https://docs.ollama.com/faq).

## Current validation status

Validated locally on September 13, 2026 with Docker Desktop Linux containers:

- Image build and dependency compatibility check passed.
- All 40 unit and API contract tests passed inside Linux.
- Ingestion populated the container volume with 30 synthetic tickets.
- Retrieval matched the native baseline: Recall@3 100% (11/11), MRR 0.955,
  Data Sync MRR 0.833, and unsupported abstentions 2/2.
- Live liveness and readiness endpoints succeeded, including host Ollama connectivity.
- Live password-reset inference cited only SR001 and returned three evidence records;
  the unsupported organization-logo question abstained.
- Recreating the API container preserved all 30 indexed tickets.
- Non-root execution and exclusion of local notes, Git metadata, environment files,
  and the host virtual environment were verified inside the container.

An earlier Docker Desktop startup failure involved inaccessible Windows runtime
sockets in the inference manager and secrets engine. Restarting Windows restored
engine availability. No factory reset was needed. The precise cause of the
Windows access failure was not established; it was outside the application.
