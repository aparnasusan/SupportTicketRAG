"""HTTP contracts use mocked dependencies; no model downloads or inference."""
import io
import json
import logging
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import api
from config import Settings
from errors import IndexUnavailable, OllamaUnavailable, InvalidModelResponse, RetrievalUnavailable
from search import RetrievedTicket

SAMPLE_TICKET = RetrievedTicket(
    ticket_id="SR001", product="Identity Hub",
    issue="Users cannot sign in after resetting a password",
    resolution="Clear the stale password-reset session.", similarity_score=0.61,
)
ISSUE = "I cannot sign in after resetting my password"
ORIGIN = "http://localhost:3000"


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.settings = Settings(_env_file=None, cors_allowed_origins=[ORIGIN])
        self.client = TestClient(api.create_app(self.settings), raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    @patch("api.check_readiness", side_effect=AssertionError("must not probe"))
    def test_liveness_is_independent(self, probe):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertTrue(response.headers["X-Request-ID"])
        probe.assert_not_called()

    @patch("api.generate_resolution")
    def test_resolution_uses_server_model_and_returns_evidence(self, generate):
        generate.return_value = ("Grounded answer", [SAMPLE_TICKET])
        response = self.client.post("/v1/resolutions", json={"issue": ISSUE, "top_k": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Grounded answer")
        self.assertTrue(response.json()["evidence_sufficient"])
        self.assertEqual(response.json()["evidence"][0]["ticket_id"], "SR001")
        generate.assert_called_once_with(ISSUE, top_k=1, model=self.settings.ollama_model,
                                         settings=self.settings)

    @patch("api.generate_resolution")
    def test_validation_boundaries(self, generate):
        for body in [
            {}, {"issue": "     "}, {"issue": "  ab  "}, {"issue": "x" * 2001},
            {"issue": ISSUE, "top_k": 0}, {"issue": ISSUE, "top_k": 6},
            {"issue": ISSUE, "model": "other-model"}, {"issue": None},
        ]:
            with self.subTest(body=body):
                response = self.client.post("/v1/resolutions", json=body)
                self.assertEqual(response.status_code, 422)
                self.assertTrue(response.headers["X-Request-ID"])
        generate.assert_not_called()

    @patch("api.generate_resolution", return_value=("answer", [SAMPLE_TICKET]))
    def test_trimmed_valid_boundaries(self, generate):
        for issue, top_k in [("12345", 1), ("x" * 2000, 5)]:
            response = self.client.post("/v1/resolutions", json={"issue": " " + issue + " ", "top_k": top_k})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(generate.call_args.args[0], issue)

    @patch("api.generate_resolution")
    def test_known_failures_are_safe_503(self, generate):
        for error in [IndexUnavailable(), RetrievalUnavailable(), OllamaUnavailable(), InvalidModelResponse()]:
            with self.subTest(error=type(error).__name__):
                generate.side_effect = error
                response = self.client.post("/v1/resolutions", json={"issue": ISSUE},
                                            headers={"Origin": ORIGIN, "X-Request-ID": "test-failure"})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.headers["X-Request-ID"], "test-failure")
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], ORIGIN)
                self.assertNotIn("Traceback", response.text)

    @patch("api.generate_resolution", side_effect=RuntimeError("PRIVATE-UPSTREAM-CONTENT"))
    def test_unexpected_errors_and_logs_are_private(self, generate):
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        handler.setFormatter(api.JsonFormatter())
        api.LOGGER.addHandler(handler)
        try:
            response = self.client.post("/v1/resolutions", json={"issue": "PRIVATE-ISSUE-CONTENT"},
                                        headers={"Origin": ORIGIN, "X-Request-ID": "correlation-1"})
        finally:
            api.LOGGER.removeHandler(handler)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal server error."})
        self.assertEqual(response.headers["X-Request-ID"], "correlation-1")
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], ORIGIN)
        for secret in ["PRIVATE-UPSTREAM-CONTENT", "PRIVATE-ISSUE-CONTENT"]:
            self.assertNotIn(secret, response.text + output.getvalue())
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r["event"] for r in records], ["request_failed", "request_completed"])
        self.assertTrue(all(r["request_id"] == "correlation-1" for r in records))

    def test_validation_does_not_echo_input_or_unknown_keys(self):
        secret = "PRIVATE-VALIDATION-CONTENT"
        response = self.client.post("/v1/resolutions", json={"issue": ISSUE, secret: secret})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(secret, response.text)
        response = self.client.post("/v1/resolutions", content='{"issue":"PRIVATE-VALIDATION-CONTENT"',
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn(secret, response.text)

    def test_request_id_policy(self):
        for value in ["valid-id_123", "", "a" * 65, "contains spaces"]:
            response = self.client.get("/health", headers={"X-Request-ID": value})
            returned = response.headers["X-Request-ID"]
            if value == "valid-id_123":
                self.assertEqual(returned, value)
            else:
                self.assertRegex(returned, r"^[a-f0-9]{32}$")

    @patch("api.check_readiness")
    def test_readiness_status_and_components(self, probe):
        for component in [None, "chroma", "ollama"]:
            components = {"chroma": {"status": "ok"}, "ollama": {"status": "ok"}}
            if component:
                components[component] = {"status": "unavailable", "reason": "probe_failed"}
            result = {"status": "not_ready" if component else "ready", "components": components}
            probe.return_value = result
            response = self.client.get("/health/ready")
            self.assertEqual(response.status_code, 503 if component else 200)
            self.assertEqual(response.json(), result)
            self.assertTrue(response.headers["X-Request-ID"])

    def test_cors_allowlist_and_preflight(self):
        headers = {"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
                   "Access-Control-Request-Headers": "content-type,x-request-id"}
        response = self.client.options("/v1/resolutions", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], ORIGIN)
        self.assertTrue(response.headers["X-Request-ID"])
        self.assertNotIn("Access-Control-Allow-Credentials", response.headers)
        for override in [
            {"Origin": "https://untrusted.example"},
            {"Access-Control-Request-Method": "DELETE"},
            {"Access-Control-Request-Headers": "x-unapproved"},
        ]:
            response = self.client.options("/v1/resolutions", headers=headers | override)
            self.assertEqual(response.status_code, 400)
            self.assertTrue(response.headers["X-Request-ID"])
        response = self.client.get("/health", headers={"Origin": "https://untrusted.example"})
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)
        response = self.client.get("/health", headers={"Origin": ORIGIN})
        self.assertIn("X-Request-ID", response.headers["Access-Control-Expose-Headers"])

    def test_empty_cors_allowlist(self):
        with TestClient(api.create_app(Settings(_env_file=None))) as client:
            response = client.get("/health", headers={"Origin": ORIGIN})
            self.assertNotIn("Access-Control-Allow-Origin", response.headers)

    @patch("api.generate_resolution")
    def test_abstention_is_a_successful_response(self, generate):
        weak = RetrievedTicket("SR001", "product", "issue", "resolution", 0.40)
        generate.return_value = ("Insufficient evidence", [weak])
        response = self.client.post("/v1/resolutions", json={"issue": ISSUE})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["evidence_sufficient"])

    def test_logging_formatter_preserves_zero_and_uses_utc(self):
        record = logging.LogRecord("test", logging.INFO, "", 0, "request_completed", (), None)
        record.created = 0
        record.duration_ms = 0
        payload = json.loads(api.JsonFormatter().format(record))
        self.assertEqual(payload["duration_ms"], 0)
        self.assertEqual(payload["timestamp"], "1970-01-01T00:00:00Z")
