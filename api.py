"""HTTP API for the Support Ticket RAG Assistant.

The API is intentionally thin: it validates HTTP requests and maps the existing
retrieval-and-generation service into structured JSON responses.
"""

from __future__ import annotations

import json
import logging
import os
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag import DEFAULT_OLLAMA_MODEL, generate_resolution
from search import RetrievedTicket, has_sufficient_evidence

app = FastAPI(
    title="Support Ticket RAG API",
    version="1.0.0",
    description=(
        "Retrieve historical support-ticket evidence and generate a grounded "
        "suggested resolution using a local Ollama model."
    ),
)
SERVER_MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


class JsonFormatter(logging.Formatter):
    """Emit machine-readable operational logs without request-body content."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%SZ"),
        }
        for field in ("event", "request_id", "method", "path", "status_code", "duration_ms", "reason"):
            if value := getattr(record, field, None):
                payload[field] = value
        return json.dumps(payload)


def configure_logger() -> logging.Logger:
    """Configure a dedicated API logger once, avoiding duplicate reload handlers."""
    logger = logging.getLogger("support_ticket_rag.api")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


LOGGER = configure_logger()


class ResolutionRequest(BaseModel):
    """Validated input accepted by the resolution endpoint."""

    model_config = ConfigDict(extra="forbid")

    issue: str = Field(
        min_length=5,
        max_length=2_000,
        description="Natural-language description of the customer's support issue.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of historical tickets to retrieve as evidence.",
    )
    @field_validator("issue")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Trim user input and reject values that contain only whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class TicketEvidence(BaseModel):
    """A retrieved ticket returned with the generated response for auditability."""

    ticket_id: str
    product: str
    issue: str
    resolution: str
    similarity_score: float


class ResolutionResponse(BaseModel):
    """Grounded answer and the evidence used to produce it."""

    answer: str
    evidence_sufficient: bool
    evidence: list[TicketEvidence]


class HealthResponse(BaseModel):
    """Basic liveness response for a local development deployment."""

    status: str


@app.middleware("http")
async def log_request(request: Request, call_next) -> Response:
    """Attach a request ID and emit a structured, privacy-conscious completion log."""
    request_id = request.headers.get("X-Request-ID", uuid4().hex)
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception(
            "request_failed",
            extra={
                "event": "request_failed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        raise

    duration_ms = round((time.perf_counter() - started_at) * 1_000, 1)
    response.headers["X-Request-ID"] = request_id
    LOGGER.info(
        "request_completed",
        extra={
            "event": "request_completed",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


def to_ticket_evidence(ticket: RetrievedTicket) -> TicketEvidence:
    """Convert the internal retrieval object into an explicit API contract."""
    return TicketEvidence(
        ticket_id=ticket.ticket_id,
        product=ticket.product,
        issue=ticket.issue,
        resolution=ticket.resolution,
        similarity_score=ticket.similarity_score,
    )


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health_check() -> HealthResponse:
    """Confirm that the web service itself is running.

    This is deliberately a lightweight liveness check. The resolution endpoint
    separately reports unavailable ChromaDB or Ollama dependencies as HTTP 503.
    """
    return HealthResponse(status="ok")


@app.post("/v1/resolutions", response_model=ResolutionResponse, tags=["resolutions"])
def create_resolution(request: ResolutionRequest) -> ResolutionResponse:
    """Return a grounded resolution and its retrieved historical-ticket evidence."""
    try:
        answer, tickets = generate_resolution(
            request.issue, top_k=request.top_k, model=SERVER_MODEL
        )
    except SystemExit as error:
        LOGGER.warning(
            "resolution_unavailable",
            extra={"event": "resolution_unavailable", "reason": str(error)},
        )
        raise HTTPException(
            status_code=503,
            detail="Resolution service is temporarily unavailable. Check the server logs.",
        ) from error

    return ResolutionResponse(
        answer=answer,
        evidence_sufficient=has_sufficient_evidence(tickets),
        evidence=[to_ticket_evidence(ticket) for ticket in tickets],
    )
