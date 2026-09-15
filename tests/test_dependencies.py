"""Readiness uses mocked I/O and never initializes an embedding model."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chromadb.errors import NotFoundError

from support_ticket_rag.config import Settings
from support_ticket_rag.dependencies import check_chroma, check_ollama, check_readiness


class DependencyTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.settings = Settings(_env_file=None, healthcheck_timeout_seconds=2)

    def test_missing_index_does_not_create_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            settings = Settings(_env_file=None, chroma_path=path)
            with patch("support_ticket_rag.dependencies.chromadb.PersistentClient") as client:
                result = check_chroma(settings)
                client.assert_not_called()
            self.assertFalse(path.exists())
            self.assertEqual(result["reason"], "index_unavailable")

    @patch("support_ticket_rag.dependencies.chromadb.PersistentClient")
    def test_existing_collection_probe(self, client):
        with patch("pathlib.Path.is_file", return_value=True):
            collection = client.return_value.get_collection.return_value
            collection.count.return_value = 30
            self.assertEqual(check_chroma(self.settings), {"status": "ok"})
            client.return_value.get_collection.assert_called_with("support_tickets")
            client.return_value.get_or_create_collection.assert_not_called()
            collection.count.return_value = 0
            self.assertEqual(check_chroma(self.settings)["reason"], "index_unavailable")
            client.return_value.get_collection.side_effect = NotFoundError("private-path")
            self.assertEqual(check_chroma(self.settings)["reason"], "index_unavailable")

    def test_storage_failures_are_sanitized(self):
        with patch("pathlib.Path.is_file", side_effect=OSError("private-path")):
            result = check_chroma(self.settings)
        self.assertEqual(result["reason"], "retrieval_unavailable")
        self.assertNotIn("private-path", json.dumps(result))

    @patch("support_ticket_rag.dependencies.urlopen")
    def test_ollama_model_and_timeout(self, opener):
        opener.return_value = io.BytesIO(json.dumps({"models": [{"name": "llama3.2:3b"}]}).encode())
        self.assertEqual(check_ollama(self.settings), {"status": "ok"})
        opener.assert_called_once_with("http://localhost:11434/api/tags", timeout=2)

    @patch("support_ticket_rag.dependencies.urlopen")
    def test_missing_model_and_malformed_responses(self, opener):
        for body, reason in [
            ({"models": []}, "model_missing"),
            ({}, "invalid_model_list"),
            ({"models": "wrong"}, "invalid_model_list"),
            ({"models": [None]}, "invalid_model_list"),
        ]:
            with self.subTest(body=body):
                opener.return_value = io.BytesIO(json.dumps(body).encode())
                self.assertEqual(check_ollama(self.settings)["reason"], reason)
        opener.return_value = io.BytesIO(b"not-json")
        self.assertEqual(check_ollama(self.settings)["reason"], "invalid_model_list")

    @patch("support_ticket_rag.dependencies.urlopen", side_effect=TimeoutError("private-url"))
    def test_ollama_timeout(self, opener):
        self.assertEqual(
            check_ollama(self.settings), {"status": "unavailable", "reason": "ollama_unreachable"}
        )

    @patch("support_ticket_rag.dependencies.urlopen")
    def test_default_latest_tag(self, opener):
        opener.return_value = io.BytesIO(b'{"models":[{"name":"local-model:latest"}]}')
        settings = Settings(_env_file=None, ollama_model="local-model")
        self.assertEqual(check_ollama(settings)["status"], "ok")

    @patch("support_ticket_rag.dependencies.check_ollama", return_value={"status": "ok"})
    @patch(
        "support_ticket_rag.dependencies.check_chroma",
        return_value={"status": "unavailable", "reason": "index_unavailable"},
    )
    def test_both_components_checked_when_one_fails(self, chroma, ollama):
        result = check_readiness(self.settings)
        self.assertEqual(result["status"], "not_ready")
        chroma.assert_called_once()
        ollama.assert_called_once()
