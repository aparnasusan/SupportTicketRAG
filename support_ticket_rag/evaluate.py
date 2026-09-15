"""Run repeatable retrieval checks against the labeled evaluation set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from support_ticket_rag.config import PROJECT_DIR
from support_ticket_rag.errors import ServiceError
from support_ticket_rag.search import has_sufficient_evidence, retrieve_tickets
from support_ticket_rag.validation import validate_issue, validate_top_k

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
        cases = json.load(file)
    validate_cases(cases)
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> None:
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation cases must be a nonempty list.")
    seen = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not {"id", "query", "category", "expected_ticket_id"} <= case.keys()
        ):
            raise ValueError(
                "Each evaluation case must include id, query, category, and expected_ticket_id."
            )
        if any(
            not isinstance(case[key], str) or not case[key].strip() for key in ("id", "category")
        ):
            raise ValueError("Evaluation case IDs and categories must be nonblank text.")
        if case["id"] in seen:
            raise ValueError("Evaluation case IDs must be unique.")
        seen.add(case["id"])
        validate_issue(case["query"])
        expected = case["expected_ticket_id"]
        if expected is not None and (not isinstance(expected, str) or not expected.strip()):
            raise ValueError("Expected ticket must be a nonblank ID or null for abstention.")


def evaluate(cases: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    """Print Recall@K, MRR, abstention behavior, and per-case evidence."""
    validate_cases(cases)
    validate_top_k(top_k)
    supported_cases = [case for case in cases if case["expected_ticket_id"]]
    category_results: dict[str, list[float]] = defaultdict(list)
    recall_hits = 0
    reciprocal_ranks: list[float] = []
    unsupported_correct = 0
    false_abstentions = 0

    for case in cases:
        tickets = retrieve_tickets(case["query"], top_k=top_k)
        ticket_ids = [ticket.ticket_id for ticket in tickets]
        strongest_score = tickets[0].similarity_score if tickets else 0.0

        if expected_ticket_id := case["expected_ticket_id"]:
            rr = reciprocal_rank(ticket_ids, expected_ticket_id)
            recall_hits += int(rr > 0)
            reciprocal_ranks.append(rr)
            category_results[case["category"]].append(rr)
            false_abstention = not has_sufficient_evidence(tickets)
            false_abstentions += int(false_abstention)
            outcome = "FALSE_ABSTENTION" if false_abstention else ("PASS" if rr else "MISS")
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
    print(f"Supported false abstentions: {false_abstentions}/{len(supported_cases)}")
    print(
        "Unsupported abstentions: "
        f"{unsupported_correct}/{unsupported_cases} "
        "(a pass means the retrieval gate stopped generation)"
    )
    print("\nMRR by supported category")
    for category, scores in sorted(category_results.items()):
        print(f"{category}: {sum(scores) / len(scores):.3f} ({len(scores)} cases)")
    return {
        "top_k": top_k,
        "supported_count": len(supported_cases),
        "recall": recall_at_k,
        "mrr": mrr,
        "supported_false_abstentions": false_abstentions,
        "unsupported_count": unsupported_cases,
        "unsupported_correct": unsupported_correct,
        "category_mrr": {
            category: sum(scores) / len(scores) for category, scores in category_results.items()
        },
    }


def passes_regression_check(result: dict[str, Any], min_mrr: float = 0.0) -> bool:
    """Require coverage and correct gating; optionally enforce an MRR floor."""
    return (
        result["supported_count"] > 0
        and result["unsupported_count"] > 0
        and result["recall"] == 1.0
        and result["mrr"] >= min_mrr
        and result["supported_false_abstentions"] == 0
        and result["unsupported_correct"] == result["unsupported_count"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ticket retrieval against labeled cases.")
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of tickets to retrieve per case."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero for misses, false abstentions, or unsupported generation.",
    )
    parser.add_argument(
        "--min-mrr", type=float, default=0.0, help="Optional MRR floor for --check (0 to 1)."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to the labeled evaluation cases JSON file.",
    )
    args = parser.parse_args()
    if not 0 <= args.min_mrr <= 1:
        parser.error("--min-mrr must be between 0 and 1.")
    try:
        result = evaluate(load_cases(args.cases), top_k=args.top_k)
    except (ValueError, ServiceError) as error:
        parser.exit(1, f"{error}\n")
    if args.check and not passes_regression_check(result, args.min_mrr):
        parser.exit(1, "Retrieval regression check failed.\n")
