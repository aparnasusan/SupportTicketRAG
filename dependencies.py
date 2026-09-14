"""Lightweight readiness probes: no ingestion, embeddings, or generation."""
import json
from http.client import HTTPException as HTTPProtocolError
import sqlite3
from urllib.error import URLError
from urllib.request import urlopen

import chromadb
from chromadb.errors import ChromaError, NotFoundError

from config import Settings
from errors import IndexUnavailable, RetrievalUnavailable, ServiceError


def open_ticket_collection(settings: Settings):
    """Open only an existing persistent index, never create a collection."""
    try:
        if not (settings.chroma_path / "chroma.sqlite3").is_file():
            raise IndexUnavailable()
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
        collection = client.get_collection(settings.chroma_collection)
        if collection.count() == 0:
            raise IndexUnavailable()
        return collection
    except NotFoundError as error:
        raise IndexUnavailable() from error
    except (ChromaError, OSError, sqlite3.Error) as error:
        raise RetrievalUnavailable() from error


def check_chroma(settings: Settings) -> dict[str, str]:
    try:
        open_ticket_collection(settings)
        return {"status": "ok"}
    except ServiceError as error:
        return {"status": "unavailable", "reason": error.code}
    except Exception:
        # Probe boundary: unknown probe failures must not report readiness.
        return {"status": "unavailable", "reason": "probe_failed"}


def check_ollama(settings: Settings) -> dict[str, str]:
    try:
        with urlopen(
            settings.ollama_base_url + "/api/tags",
            timeout=settings.healthcheck_timeout_seconds,
        ) as response:
            body = json.load(response)
        models = body["models"]
        if not isinstance(models, list) or any(
            not isinstance(model, dict) or not isinstance(model.get("name"), str)
            for model in models
        ):
            raise ValueError("Invalid model list")
        names = {model["name"] for model in models}
        # Ollama omits :latest in user input but reports it in model listings.
        name = settings.ollama_model
        expected = name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"
        if name not in names and expected not in names:
            return {"status": "unavailable", "reason": "model_missing"}
        return {"status": "ok"}
    except (URLError, OSError, HTTPProtocolError):
        return {"status": "unavailable", "reason": "ollama_unreachable"}
    except (ValueError, KeyError, TypeError):
        return {"status": "unavailable", "reason": "invalid_model_list"}
    except Exception:
        return {"status": "unavailable", "reason": "probe_failed"}


def check_readiness(settings: Settings) -> dict:
    components = {"chroma": check_chroma(settings), "ollama": check_ollama(settings)}
    ready = all(component["status"] == "ok" for component in components.values())
    return {"status": "ready" if ready else "not_ready", "components": components}
