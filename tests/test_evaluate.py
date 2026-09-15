"""Retrieval ranking and gating are distinct evaluation outcomes."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from support_ticket_rag.evaluate import evaluate, passes_regression_check, validate_cases
from support_ticket_rag.search import RetrievedTicket

CASES = [
    {"id": "one", "category": "auth", "query": "Cannot sign in", "expected_ticket_id": "SR001"},
    {
        "id": "two",
        "category": "unsupported",
        "query": "Change company logo",
        "expected_ticket_id": None,
    },
]


def ticket(score, name="SR001"):
    return RetrievedTicket(name, "product", "issue", "resolution", score)


class EvaluationTests(unittest.TestCase):
    def run_evaluation(self, responses):
        with (
            patch("support_ticket_rag.evaluate.retrieve_tickets", side_effect=responses),
            redirect_stdout(io.StringIO()) as output,
        ):
            result = evaluate(CASES, 3)
        return result, output.getvalue()

    def test_correct_rank_can_still_be_a_false_abstention(self):
        result, output = self.run_evaluation([[ticket(0.49)], [ticket(0.2)]])
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["mrr"], 1.0)
        self.assertEqual(result["supported_false_abstentions"], 1)
        self.assertIn("FALSE_ABSTENTION", output)
        self.assertFalse(passes_regression_check(result))

    def test_expected_second_rank_is_preserved_and_floor_is_enforced(self):
        result, _ = self.run_evaluation([[ticket(0.7, "SR002"), ticket(0.6)], [ticket(0.2)]])
        self.assertEqual(result["mrr"], 0.5)
        self.assertTrue(passes_regression_check(result, 0.5))
        self.assertFalse(passes_regression_check(result, 0.9))

    def test_unsupported_generation_and_missing_expected_ticket_fail(self):
        for responses in [
            [[ticket(0.7)], [ticket(0.6)]],
            [[ticket(0.7, "SR002")], [ticket(0.2)]],
            [[], []],
        ]:
            result, _ = self.run_evaluation(responses)
            self.assertFalse(passes_regression_check(result))

    def test_case_validation_rejects_bad_or_empty_inputs(self):
        for cases in [
            [],
            {},
            [CASES[0], CASES[0]],
            [{}],
            [CASES[0] | {"query": ""}],
            [CASES[0] | {"expected_ticket_id": ""}],
        ]:
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                validate_cases(cases)

    def test_invalid_top_k_never_retrieves(self):
        with patch("support_ticket_rag.evaluate.retrieve_tickets") as retrieve:
            with self.assertRaises(ValueError):
                evaluate(CASES, 0)
            retrieve.assert_not_called()
