"""Settings checks isolated from the developer's environment and .env."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from support_ticket_rag.config import PROJECT_DIR, ConfigurationError, Settings, get_settings


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        get_settings.cache_clear()
        self.addCleanup(get_settings.cache_clear)

    def test_defaults_and_project_relative_paths(self):
        settings = Settings(_env_file=None)
        self.assertEqual(PROJECT_DIR, Path(__file__).resolve().parent.parent)
        self.assertTrue((PROJECT_DIR / "data" / "tickets.csv").is_file())
        self.assertEqual(settings.ollama_model, "llama3.2:3b")
        self.assertEqual(settings.ollama_timeout_seconds, 120)
        self.assertEqual(settings.cors_allowed_origins, ())
        self.assertEqual(
            Settings(_env_file=None, chroma_path="other").chroma_path,
            (PROJECT_DIR / "other").resolve(),
        )

    def test_environment_overrides_dotenv_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text(
                'OLLAMA_MODEL=from-file\nCORS_ALLOWED_ORIGINS=["http://localhost:3000"]\n'
            )
            self.assertEqual(Settings(_env_file=dotenv).ollama_model, "from-file")
            with patch.dict(os.environ, {"OLLAMA_MODEL": "from-env"}):
                settings = Settings(_env_file=dotenv)
            self.assertEqual(settings.ollama_model, "from-env")
            self.assertEqual(settings.cors_allowed_origins, ("http://localhost:3000",))

    def test_rejects_invalid_values(self):
        for values in [
            {"ollama_model": " "},
            {"chroma_collection": " "},
            {"chroma_path": ""},
            {"ollama_timeout_seconds": 0},
            {"ollama_timeout_seconds": float("inf")},
            {"healthcheck_timeout_seconds": -1},
            {"healthcheck_timeout_seconds": float("nan")},
            {"ollama_base_url": "file:///tmp"},
            {"ollama_base_url": "http://user:secret@localhost"},
            {"ollama_base_url": "http://localhost:99999"},
            {"ollama_base_url": "http://localhost/api/chat"},
            {"cors_allowed_origins": ["*"]},
            {"cors_allowed_origins": ["https://*.example.com"]},
            {"cors_allowed_origins": ["https://example.com/path"]},
            {"cors_allowed_origins": ["https://example.com?"]},
        ]:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Settings(_env_file=None, **values)

    def test_startup_error_does_not_echo_bad_configuration(self):
        with patch.dict(
            os.environ, {"OLLAMA_BASE_URL": "http://private-user:private-secret@localhost"}
        ):
            with self.assertRaises(ConfigurationError) as caught:
                get_settings()
        self.assertNotIn("private-secret", str(caught.exception))
        self.assertIn("ollama_base_url", str(caught.exception))

    def test_malformed_json_configuration_fails_safely(self):
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "private-invalid-json"}):
            with self.assertRaises(ConfigurationError) as caught:
                get_settings()
        self.assertNotIn("private-invalid-json", str(caught.exception))

    def test_settings_are_cached_and_immutable(self):
        with patch(
            "support_ticket_rag.config.Settings", return_value=Settings(_env_file=None)
        ) as factory:
            self.assertIs(get_settings(), get_settings())
            factory.assert_called_once()
        with self.assertRaises(ValidationError):
            get_settings().ollama_model = "changed"
