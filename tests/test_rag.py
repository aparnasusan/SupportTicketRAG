"""Focused unit tests for non-model grounding checks."""

import unittest
from itertools import permutations

from support_ticket_rag.rag import (
    REQUIRED_HEADINGS,
    select_generation_evidence,
    validate_grounded_answer,
)
from support_ticket_rag.search import RetrievedTicket

TICKETS = [
    RetrievedTicket(
        ticket_id="SR001",
        product="Identity Hub",
        issue="Users cannot sign in after resetting a password",
        resolution="Clear the stale password-reset session.",
        similarity_score=0.60,
    )
]


class GroundingValidationTests(unittest.TestCase):
    def test_rejects_unknown_ticket_in_any_section(self) -> None:
        contents = ["A stale session.", "Clear the session.", "SR001", "Limited evidence."]
        for section in range(len(REQUIRED_HEADINGS)):
            with self.subTest(section=REQUIRED_HEADINGS[section]):
                modified = contents.copy()
                modified[section] += " See SR999."
                answer = "\n".join(
                    f"{heading}\n{content}" for heading, content in zip(REQUIRED_HEADINGS, modified)
                )
                self.assertIn("not retrieved", validate_grounded_answer(answer, TICKETS))

    def test_rejects_every_incorrect_heading_order(self) -> None:
        for headings in permutations(REQUIRED_HEADINGS):
            if headings == REQUIRED_HEADINGS:
                continue
            with self.subTest(headings=headings):
                answer = "\n".join(f"{heading}\nContent." for heading in headings)
                self.assertIn("order", validate_grounded_answer(answer, TICKETS))

    def test_rejects_empty_or_whitespace_only_sections(self) -> None:
        for section in range(len(REQUIRED_HEADINGS)):
            for empty in ["", " \t\n "]:
                with self.subTest(section=REQUIRED_HEADINGS[section], empty=empty):
                    answer = "\n".join(
                        f"{heading}\n{empty if index == section else 'Content.'}"
                        for index, heading in enumerate(REQUIRED_HEADINGS)
                    )
                    self.assertIn("nonempty", validate_grounded_answer(answer, TICKETS))

    def test_accepts_supported_prose_references_and_explicit_no_citations(self) -> None:
        for contents in [
            [
                "SR001 describes a stale session.",
                "Follow SR001: clear the session.",
                "SR001",
                "Limited evidence.",
            ],
            ["Unknown.", "Gather more details.", "None.", "No direct evidence."],
        ]:
            with self.subTest(contents=contents):
                answer = "\n".join(
                    f"{heading}\n{content}" for heading, content in zip(REQUIRED_HEADINGS, contents)
                )
                self.assertIsNone(validate_grounded_answer(answer, TICKETS))

    def test_uses_only_a_clear_high_confidence_match_for_generation(self) -> None:
        related_ticket = RetrievedTicket(
            ticket_id="SR002",
            product="Identity Hub",
            issue="An account is locked after repeated attempts",
            resolution="Unlock the account.",
            similarity_score=0.54,
        )
        selected = select_generation_evidence([TICKETS[0], related_ticket])
        self.assertEqual([ticket.ticket_id for ticket in selected], ["SR001"])

    def test_keeps_multiple_candidates_when_retrieval_is_ambiguous(self) -> None:
        close_ticket = RetrievedTicket(
            ticket_id="SR002",
            product="Identity Hub",
            issue="An account is locked after repeated attempts",
            resolution="Unlock the account.",
            similarity_score=0.59,
        )
        selected = select_generation_evidence([TICKETS[0], close_ticket])
        self.assertEqual([ticket.ticket_id for ticket in selected], ["SR001", "SR002"])

    def test_accepts_consistent_direct_evidence(self) -> None:
        answer = """Likely Cause
A stale password-reset session.
Suggested Resolution
Clear the stale session.
Relevant Historical Ticket IDs
SR001
Evidence Limitations
No material limitation in the retrieved evidence."""
        self.assertIsNone(validate_grounded_answer(answer, TICKETS))

    def test_rejects_evidence_contradiction(self) -> None:
        answer = """Likely Cause
A stale password-reset session.
Suggested Resolution
Clear the stale session.
Relevant Historical Ticket IDs
SR001
Evidence Limitations
There is no direct evidence for this resolution."""
        self.assertIsNotNone(validate_grounded_answer(answer, TICKETS))

    def test_rejects_unknown_citation(self) -> None:
        answer = """Likely Cause
Unknown.
Suggested Resolution
Investigate.
Relevant Historical Ticket IDs
SR999
Evidence Limitations
Limited evidence."""
        self.assertIsNotNone(validate_grounded_answer(answer, TICKETS))


if __name__ == "__main__":
    unittest.main()
