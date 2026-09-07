"""Generate a grounded support resolution using local Ollama inference."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from search import RetrievedTicket, has_sufficient_evidence, retrieve_tickets

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_TEMPERATURE = 0
DEFAULT_SEED = 42
# A high-confidence match can be used as the sole generation source while the
# complete retrieved list remains available for audit in the API and UI.
HIGH_CONFIDENCE_SCORE = 0.60
HIGH_CONFIDENCE_MARGIN = 0.03
REQUIRED_HEADINGS = (
    "Likely Cause",
    "Suggested Resolution",
    "Relevant Historical Ticket IDs",
    "Evidence Limitations",
)


@dataclass
class OllamaClient:
    """Minimal client for Ollama's local chat API.

    Keeping this client separate from retrieval makes a future hosted provider
    replacement a small, isolated change.
    """

    model: str
    url: str = DEFAULT_OLLAMA_URL

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                # Stable local generation makes quality checks reproducible.
                "options": {"temperature": DEFAULT_TEMPERATURE, "seed": DEFAULT_SEED},
            }
        ).encode("utf-8")
        request = Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                body: dict[str, Any] = json.load(response)
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise SystemExit(f"Ollama returned HTTP {error.code}: {details}") from error
        except URLError as error:
            raise SystemExit(
                "Cannot reach Ollama. Install Ollama, start it, and run "
                f"`ollama pull {self.model}` before using rag.py. ({error.reason})"
            ) from error

        try:
            return str(body["message"]["content"]).strip()
        except (KeyError, TypeError) as error:
            raise SystemExit(f"Unexpected Ollama response: {body}") from error


def format_evidence(tickets: list[RetrievedTicket]) -> str:
    """Format retrieved records as the only evidence allowed in the response."""
    if not tickets:
        return "No historical tickets were retrieved."

    return "\n\n".join(
        (
            f"Historical ticket {ticket.ticket_id}\n"
            f"Product: {ticket.product}\n"
            f"Issue: {ticket.issue}\n"
            f"Resolution: {ticket.resolution}"
        )
        for ticket in tickets
    )


def select_generation_evidence(tickets: list[RetrievedTicket]) -> list[RetrievedTicket]:
    """Select the evidence sent to the LLM without hiding retrieved alternatives.

    When the strongest record passes the high-confidence score and clears the
    runner-up by the configured margin, it is the sole source supplied to the
    model. This prevents merely related records from being cited as if they were
    direct support. Ambiguous retrievals continue to provide all candidates.
    """
    if not tickets:
        return []

    strongest = tickets[0]
    runner_up_score = tickets[1].similarity_score if len(tickets) > 1 else 0.0
    if (
        strongest.similarity_score >= HIGH_CONFIDENCE_SCORE
        and strongest.similarity_score - runner_up_score >= HIGH_CONFIDENCE_MARGIN
    ):
        return [strongest]
    return tickets


def build_messages(user_issue: str, tickets: list[RetrievedTicket]) -> list[dict[str, str]]:
    """Build a grounding-first prompt for the local LLM."""
    system_prompt = """You are a technical support resolution assistant.
Use only the historical tickets provided as evidence. Do not invent product behavior,
procedures, or causes that are not supported by that evidence. If the evidence is weak
or does not support a confident answer, say so explicitly.

Follow these evidence rules:
- A ticket is a direct match when it describes the same issue and provides a
  resolution that applies to the user's issue. State a direct match confidently.
- Cite only ticket IDs that directly support the likely cause or suggested resolution.
  Do not cite tickets merely because they are topically related.
- In Evidence Limitations, name only a genuine gap in the retrieved evidence.
  Never claim a limitation that contradicts a ticket's issue or resolution.
- Never say there is no direct evidence when you cite a ticket as direct support.
  If a ticket directly supports the resolution, identify only that ticket rather
  than adding merely related tickets.
- If a direct matching ticket exists, write: "No material limitation in the retrieved
  evidence." Do not add unsupported uncertainty.

Start immediately with, and return exactly, these headings:
Likely Cause
Suggested Resolution
Relevant Historical Ticket IDs
Evidence Limitations
"""
    user_prompt = (
        f"User issue:\n{user_issue}\n\n"
        f"Retrieved historical tickets:\n{format_evidence(tickets)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_answer_section(answer: str, heading: str, next_heading: str | None = None) -> str:
    """Return the text below one required heading in a generated answer."""
    start = answer.find(heading)
    if start == -1:
        return ""
    start += len(heading)
    end = answer.find(next_heading, start) if next_heading else len(answer)
    return answer[start:end if end != -1 else len(answer)].strip()


def validate_grounded_answer(answer: str, tickets: list[RetrievedTicket]) -> str | None:
    """Return a repair reason when a response violates basic evidence constraints.

    This is intentionally narrow. It checks answer structure, prohibits citations
    outside retrieved evidence, and catches a clear contradiction where an answer
    cites ticket evidence while claiming that no direct evidence exists. It does
    not attempt to decide semantic correctness in code.
    """
    if any(answer.count(heading) != 1 for heading in REQUIRED_HEADINGS):
        return "The response must include each required heading exactly once."

    citations = extract_answer_section(
        answer, "Relevant Historical Ticket IDs", "Evidence Limitations"
    )
    cited_ids = set(re.findall(r"\bSR\d{3}\b", citations))
    retrieved_ids = {ticket.ticket_id for ticket in tickets}
    if unknown_ids := cited_ids.difference(retrieved_ids):
        return f"The response cited ticket IDs that were not retrieved: {sorted(unknown_ids)}."

    limitations = extract_answer_section(answer, "Evidence Limitations").lower()
    contradiction_phrases = ("no direct evidence", "no evidence", "without evidence")
    if cited_ids and any(phrase in limitations for phrase in contradiction_phrases):
        return (
            "The response cites ticket evidence but says there is no direct evidence. "
            "Make the citation and limitation internally consistent."
        )
    return None


def build_repair_messages(
    user_issue: str,
    tickets: list[RetrievedTicket],
    invalid_answer: str,
    repair_reason: str,
) -> list[dict[str, str]]:
    """Ask the model to repair a specific grounding violation using the same evidence."""
    messages = build_messages(user_issue, tickets)
    messages.append(
        {
            "role": "assistant",
            "content": invalid_answer,
        }
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "Repair the prior answer using only the retrieved tickets. "
                f"Problem: {repair_reason} Return the four required headings only."
            ),
        }
    )
    return messages


def print_evidence(tickets: list[RetrievedTicket]) -> None:
    print("\nRetrieved evidence:")
    for ticket in tickets:
        print(f"- {ticket.ticket_id} ({ticket.product}, similarity {ticket.similarity_score:.3f})")


def generate_resolution(
    user_issue: str, top_k: int, model: str
) -> tuple[str, list[RetrievedTicket]]:
    """Retrieve evidence and return a grounded local-model response."""
    tickets = retrieve_tickets(user_issue, top_k=top_k)
    if not tickets:
        raise SystemExit("No historical tickets were retrieved; no answer was generated.")

    if not has_sufficient_evidence(tickets):
        return (
            "Likely Cause\n"
            "The available historical tickets do not provide sufficiently relevant evidence.\n\n"
            "Suggested Resolution\n"
            "Do not provide a specific remediation yet. Gather more details or route the issue "
            "to the appropriate support team.\n\n"
            "Relevant Historical Ticket IDs\n"
            "None.\n\n"
            "Evidence Limitations\n"
            "The closest retrieved ticket did not meet the minimum relevance threshold, so "
            "the local model was not asked to generate a resolution.",
            tickets,
        )

    client = OllamaClient(model=model)
    generation_tickets = select_generation_evidence(tickets)
    answer = client.chat(build_messages(user_issue, generation_tickets))
    if repair_reason := validate_grounded_answer(answer, generation_tickets):
        answer = client.chat(
            build_repair_messages(user_issue, generation_tickets, answer, repair_reason)
        )
        if repair_reason := validate_grounded_answer(answer, generation_tickets):
            raise SystemExit(f"The local model returned an invalid grounded response: {repair_reason}")
    return answer, tickets


def run_rag(user_issue: str, top_k: int, model: str) -> None:
    """Run the command-line RAG experience."""
    answer, tickets = generate_resolution(user_issue, top_k=top_k, model=model)
    print(f"\nUser issue: {user_issue}\n")
    print("Evidence-grounded suggested resolution:\n")
    print(answer)
    print_evidence(tickets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a grounded resolution with local Ollama.")
    parser.add_argument("issue", help="Natural-language description of the support issue.")
    parser.add_argument("--top-k", type=int, default=3, help="Tickets to provide as evidence.")
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        help=f"Ollama model to use (default: {DEFAULT_OLLAMA_MODEL}).",
    )
    args = parser.parse_args()
    run_rag(args.issue, top_k=args.top_k, model=args.model)
