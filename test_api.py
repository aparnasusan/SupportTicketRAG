"""API contract tests that run without ChromaDB or Ollama inference."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from search import RetrievedTicket


SAMPLE_TICKET = RetrievedTicket(
    ticket_id="SR001",
    product="Identity Hub",
    issue="Users cannot sign in after resetting a password",
    resolution="Clear the stale password-reset session.",
    similarity_score=0.61,
)


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

    def test_health_check_returns_ok_and_request_id(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertTrue(response.headers["X-Request-ID"])

    @patch("api.generate_resolution")
    def test_resolution_uses_server_model_and_returns_evidence(self, generate_resolution) -> None:
        generate_resolution.return_value = ("Grounded answer", [SAMPLE_TICKET])

        response = self.client.post(
            "/v1/resolutions",
            json={"issue": "I cannot sign in after resetting my password", "top_k": 1},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Grounded answer")
        self.assertTrue(response.json()["evidence_sufficient"])
        self.assertEqual(response.json()["evidence"][0]["ticket_id"], "SR001")
        generate_resolution.assert_called_once_with(
            "I cannot sign in after resetting my password", top_k=1, model=api.SERVER_MODEL
        )

    def test_resolution_rejects_client_model_selection(self) -> None:
        response = self.client.post(
            "/v1/resolutions",
            json={"issue": "I cannot sign in after resetting my password", "model": "other-model"},
        )
        self.assertEqual(response.status_code, 422)

    def test_resolution_rejects_blank_issue(self) -> None:
        response = self.client.post("/v1/resolutions", json={"issue": "     "})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
