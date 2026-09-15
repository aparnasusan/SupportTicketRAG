"""Service boundaries and repair behavior, without live inference."""

import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from support_ticket_rag.config import Settings
from support_ticket_rag.errors import IndexUnavailable, InvalidModelResponse, OllamaUnavailable
from support_ticket_rag.rag import OllamaClient, generate_resolution
from support_ticket_rag.search import RetrievedTicket

TICKET = RetrievedTicket("SR001", "Identity Hub", "stale session", "Clear the session", 0.65)
ANSWER = """Likely Cause
A stale session.
Suggested Resolution
Clear the session.
Relevant Historical Ticket IDs
SR001
Evidence Limitations
No material limitation in the retrieved evidence."""


class ServiceTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.settings = Settings(
            _env_file=None, ollama_base_url="http://localhost:12345", ollama_timeout_seconds=17
        )

    @patch("support_ticket_rag.rag.urlopen")
    def test_ollama_configuration_and_deterministic_payload(self, opener):
        opener.return_value = io.BytesIO(b'{"message":{"content":" answer "}}')
        answer = OllamaClient("test-model", self.settings).chat(
            [{"role": "user", "content": "issue"}]
        )
        self.assertEqual(answer, "answer")
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:12345/api/chat")
        self.assertEqual(opener.call_args.kwargs["timeout"], 17)
        payload = json.loads(request.data)
        self.assertEqual(payload["options"], {"temperature": 0, "seed": 42})
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "test-model")

    @patch("support_ticket_rag.rag.urlopen")
    def test_bad_ollama_responses_are_sanitized(self, opener):
        for raw in [
            b"PRIVATE-BAD-JSON",
            b"null",
            b"[]",
            b"{}",
            b'{"message":null}',
            b'{"message":{"content":42}}',
            b'{"message":{"content":""}}',
        ]:
            with self.subTest(raw=raw):
                opener.return_value = io.BytesIO(raw)
                with self.assertRaises(InvalidModelResponse) as caught:
                    OllamaClient("test", self.settings).chat([])
                self.assertNotIn("PRIVATE", str(caught.exception))

    @patch("support_ticket_rag.rag.urlopen")
    def test_network_failures_are_safe(self, opener):
        for error in [
            TimeoutError("PRIVATE"),
            URLError("PRIVATE"),
            HTTPError("http://localhost", 500, "PRIVATE", {}, io.BytesIO(b"PRIVATE")),
        ]:
            with self.subTest(error=type(error).__name__):
                opener.side_effect = error
                with self.assertRaises(OllamaUnavailable) as caught:
                    OllamaClient("test", self.settings).chat([])
                self.assertNotIn("PRIVATE", str(caught.exception))

    @patch("support_ticket_rag.rag.OllamaClient.chat")
    @patch("support_ticket_rag.rag.retrieve_tickets")
    def test_abstention_never_calls_ollama(self, retrieve, chat):
        retrieve.return_value = [RetrievedTicket("SR001", "p", "i", "r", 0.49)]
        answer, tickets = generate_resolution("unsupported", 3, "test", settings=self.settings)
        self.assertIn("None.", answer)
        chat.assert_not_called()
        retrieve.assert_called_once_with("unsupported", top_k=3, settings=self.settings)

    @patch("support_ticket_rag.rag.OllamaClient.chat")
    @patch("support_ticket_rag.rag.retrieve_tickets", return_value=[])
    def test_no_results_remains_unavailable(self, retrieve, chat):
        with self.assertRaises(IndexUnavailable):
            generate_resolution("issue", 3, "test", settings=self.settings)
        chat.assert_not_called()

    @patch("support_ticket_rag.rag.OllamaClient.chat", side_effect=["invalid", ANSWER])
    @patch("support_ticket_rag.rag.retrieve_tickets", return_value=[TICKET])
    def test_exactly_one_successful_repair(self, retrieve, chat):
        answer, tickets = generate_resolution("issue", 3, "test", settings=self.settings)
        self.assertEqual(answer, ANSWER)
        self.assertEqual(chat.call_count, 2)
        self.assertEqual(tickets, [TICKET])

    @patch("support_ticket_rag.rag.OllamaClient.chat", return_value="invalid")
    @patch("support_ticket_rag.rag.retrieve_tickets", return_value=[TICKET])
    def test_second_invalid_answer_fails_without_more_retries(self, retrieve, chat):
        with self.assertRaises(InvalidModelResponse):
            generate_resolution("issue", 3, "test", settings=self.settings)
        self.assertEqual(chat.call_count, 2)

    @patch("support_ticket_rag.rag.OllamaClient.chat", return_value=ANSWER)
    @patch("support_ticket_rag.rag.retrieve_tickets")
    def test_all_retrieved_evidence_returned_but_only_direct_evidence_sent(self, retrieve, chat):
        alternative = RetrievedTicket("SR002", "p", "related", "r", 0.52)
        retrieve.return_value = [TICKET, alternative]
        answer, tickets = generate_resolution("issue", 3, "test", settings=self.settings)
        evidence_prompt = chat.call_args.args[0][1]["content"]
        self.assertIn("SR001", evidence_prompt)
        self.assertNotIn("SR002", evidence_prompt)
        self.assertEqual(tickets, [TICKET, alternative])
        chat.assert_called_once()
