"""Run repeatable retrieval checks against the labeled evaluation set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from search import has_sufficient_evidence, retrieve_tickets

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = PROJECT_DIR / "data" / "evaluation_cases.json"


def reciprocal_rank(ticket_ids: list[str], expected_ticket_id: str) -> float:
    """Return 1/rank when the expected ticket appears, otherwise zero."""
    try:
        return 1.0 / (ticket_ids.index(expected_ticket_id) + 1)
    except ValueError:
        return 0.0


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load evaluation cases from a version-controlled JSON file."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def evaluate(cases: list[dict[str, Any]], top_k: int) -> None:
    """Print Recall@K, MRR, abstention behavior, and per-case evidence."""
    supported_cases = [case for case in cases if case["expected_ticket_id"]]
    category_results: dict[str, list[float]] = defaultdict(list)
    recall_hits = 0
    reciprocal_ranks: list[float] = []
    unsupported_correct = 0

    for case in cases:
        tickets = retrieve_tickets(case["query"], top_k=top_k)
        ticket_ids = [ticket.ticket_id for ticket in tickets]
        strongest_score = tickets[0].similarity_score if tickets else 0.0

        if expected_ticket_id := case["expected_ticket_id"]:
            rr = reciprocal_rank(ticket_ids, expected_ticket_id)
            recall_hits += int(rr > 0)
            reciprocal_ranks.append(rr)
            category_results[case["category"]].append(rr)
            outcome = "PASS" if rr else "MISS"
            expected = expected_ticket_id
        else:
            abstained = not has_sufficient_evidence(tickets)
            unsupported_correct += int(abstained)
            outcome = "PASS" if abstained else "REVIEW"
            expected = "abstain"

        print(
            f"{outcome:6} {case['id']} | expected: {expected:7} | "
            f"top result: {ticket_ids[0] if ticket_ids else 'none':5} | "
            f"score: {strongest_score:.3f} | {case['query']}"
        )

    recall_at_k = recall_hits / len(supported_cases) if supported_cases else 0.0
    mrr = sum(reciprocal_ranks) / len(supported_cases) if supported_cases else 0.0
    unsupported_cases = len(cases) - len(supported_cases)

    print("\nRetrieval summary")
    print(f"Recall@{top_k}: {recall_at_k:.1%} ({recall_hits}/{len(supported_cases)})")
    print(f"MRR: {mrr:.3f}")
    print(
        "Unsupported abstentions: "
        f"{unsupported_correct}/{unsupported_cases} "
        "(a pass means the retrieval gate stopped generation)"
    )
    print("\nMRR by supported category")
    for category, scores in sorted(category_results.items()):
        print(f"{category}: {sum(scores) / len(scores):.3f} ({len(scores)} cases)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ticket retrieval against labeled cases.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of tickets to retrieve per case.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to the labeled evaluation cases JSON file.",
    )
    args = parser.parse_args()
    evaluate(load_cases(args.cases), top_k=args.top_k)
