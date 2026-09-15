# Local Docker setup

The FastAPI service runs in a non-root Linux container. Ollama runs on the Windows
host, while named volumes retain the Chroma index and embedding-model cache.
Streamlit runs separately. The API is exposed only on host loopback.

## Requirements

- Docker Desktop running Linux containers.
- Ollama running on the host with `llama3.2:3b` installed.
- Internet access and sufficient disk space for initial dependency and model downloads.

## Build and run

From the repository directory:

```powershell
ollama pull llama3.2:3b
docker compose build
docker compose run --rm --no-deps api python -m support_ticket_rag.ingest
docker compose up -d
docker compose ps
```

Ingestion downloads the embedding model on first use and leaves an existing
nonempty index intact. API startup does not build an index or download models.

Open [API documentation](http://127.0.0.1:8000/docs).

## Verify

```powershell
docker compose run --rm --no-deps api python -m unittest discover -s tests -v
docker compose run --rm --no-deps api python -m support_ticket_rag.evaluate --check --min-mrr 0.9545
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Liveness checks that the API responds. Readiness checks the populated index and
configured Ollama model; it does not run inference. Docker uses readiness for its
health status. An unhealthy status does not automatically restart the container.

## Configuration and troubleshooting

Compose connects to host Ollama through `http://host.docker.internal:11434`.
Inside a container, `localhost` refers to the container itself.

Set overrides in the project `.env` or shell environment:

| Setting | Default | Purpose |
| --- | --- | --- |
| `DOCKER_OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama address used by Compose |
| `OLLAMA_MODEL` | `llama3.2:3b` | Installed model to use |
| `SUPPORT_API_PORT` | `8000` | Host port; use another port if occupied |

Run `docker compose up -d` after configuration changes. The native application's
`OLLAMA_BASE_URL` is intentionally separate from the Docker setting.
See [OPERATIONS.md](OPERATIONS.md) for timeout, CORS, and other application settings.

If readiness fails, inspect `docker compose logs --tail 50 api`, confirm ingestion
completed, and check that Ollama is running with the configured model installed.
If Docker cannot connect to Ollama, check the host address and listener/firewall
configuration; do not expose Ollama publicly.

## Stop, update, and preserve data

```powershell
docker compose stop
docker compose start
```

After source changes, rebuild and replace the container:

```powershell
docker compose up -d --build
```

`docker compose down` removes the containers and network while retaining named
volumes. **Adding `-v` deletes the container index and embedding cache.** The native
Windows index is separate.

After intentionally changing the ticket CSV, rebuild the container index:

```powershell
docker compose stop api
docker compose build
docker compose run --rm --no-deps api python -m support_ticket_rag.ingest --reset
docker compose run --rm --no-deps api python -m support_ticket_rag.evaluate --check --min-mrr 0.9545
docker compose up -d
```

## Current validation status

The packaged Linux build passed all 61 tests, matched the recorded retrieval
baseline, and passed live supported/unsupported API checks. All 30 tickets survived
container replacement. See [README.md](README.md#evaluation-and-verification) for
metrics and evaluation limitations.
