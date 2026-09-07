"""Focused unit tests for non-model grounding checks."""

import unittest

from rag import select_generation_evidence, validate_grounded_answer
from search import RetrievedTicket


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
