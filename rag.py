"""Generate a grounded support resolution using local Ollama inference."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from search import RetrievedTicket, retrieve_tickets

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


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
            {"model": self.model, "messages": messages, "stream": False}
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


def print_evidence(tickets: list[RetrievedTicket]) -> None:
    print("\nRetrieved evidence:")
    for ticket in tickets:
        print(f"- {ticket.ticket_id} ({ticket.product}, similarity {ticket.similarity_score:.3f})")


def run_rag(user_issue: str, top_k: int, model: str) -> None:
    tickets = retrieve_tickets(user_issue, top_k=top_k)
    if not tickets:
        raise SystemExit("No historical tickets were retrieved; no answer was generated.")

    answer = OllamaClient(model=model).chat(build_messages(user_issue, tickets))
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
