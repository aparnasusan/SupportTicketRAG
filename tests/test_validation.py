"""Shared entry-point validation rejects invalid input before dependency I/O."""

import unittest
from unittest.mock import patch

from support_ticket_rag.rag import generate_resolution
from support_ticket_rag.search import retrieve_tickets
from support_ticket_rag.validation import (
    InputValidationError,
    validate_issue,
    validate_model,
    validate_top_k,
)


class InputTests(unittest.TestCase):
    def test_issue_boundaries_and_trimming(self):
        self.assertEqual(validate_issue("  hello  "), "hello")
        self.assertEqual(len(validate_issue("x" * 2000)), 2000)
        for value in ["", "abcd", "x" * 2001, None, 123]:
            with self.subTest(value=type(value)), self.assertRaises(InputValidationError):
                validate_issue(value)

    def test_top_k_requires_integer_in_range(self):
        for value in [0, 6, True, 1.0, "3", None]:
            with self.subTest(value=value), self.assertRaises(InputValidationError):
                validate_top_k(value)
        self.assertEqual(validate_top_k(1), 1)
        self.assertEqual(validate_top_k(5), 5)

    def test_model_must_be_nonblank(self):
        self.assertEqual(validate_model(" local-model "), "local-model")
        for value in ["", "  ", "model\nname", None]:
            with self.subTest(value=value), self.assertRaises(InputValidationError):
                validate_model(value)

    def test_bad_generation_inputs_do_not_retrieve(self):
        with patch("support_ticket_rag.rag.retrieve_tickets") as retrieve:
            for issue, top_k, model in [
                ("bad", 3, "model"),
                ("valid issue", 0, "model"),
                ("valid issue", 3, " "),
            ]:
                with self.assertRaises(InputValidationError):
                    generate_resolution(issue, top_k, model)
            retrieve.assert_not_called()

    def test_bad_retrieval_inputs_do_not_open_storage(self):
        with patch("support_ticket_rag.search.open_ticket_collection") as collection:
            for issue, top_k in [("bad", 3), ("valid issue", 6)]:
                with self.assertRaises(InputValidationError):
                    retrieve_tickets(issue, top_k)
            collection.assert_not_called()
