"""Thin HTTP boundary for the shared local RAG service."""
from __future__ import annotations

import json
import logging
import re
import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import Settings, get_settings
from dependencies import check_readiness
from errors import ServiceError
from rag import generate_resolution
from search import RetrievedTicket, has_sufficient_evidence


class JsonFormatter(logging.Formatter):
    """Emit only approved operational fields, never exception or body content."""
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%SZ"),
        }
        for field in ("event", "request_id", "method", "path", "status_code", "duration_ms", "reason"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload)


def configure_logger() -> logging.Logger:
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
    model_config = ConfigDict(extra="forbid")
    issue: str = Field(
        min_length=5, max_length=2_000,
        description="Natural-language description of the customer's support issue.",
    )
    top_k: int = Field(
        default=3, ge=1, le=5,
        description="Number of historical tickets to retrieve as evidence.",
    )

    @field_validator("issue", mode="before")
    @classmethod
    def trim_issue(cls, value):
        return value.strip() if isinstance(value, str) else value


class TicketEvidence(BaseModel):
    ticket_id: str
    product: str
    issue: str
    resolution: str
    similarity_score: float


class ResolutionResponse(BaseModel):
    answer: str
    evidence_sufficient: bool
    evidence: list[TicketEvidence]


class HealthResponse(BaseModel):
    status: str


class ComponentHealth(BaseModel):
    status: str
    reason: str | None = None


class ReadinessResponse(BaseModel):
    status: str
    components: dict[str, ComponentHealth]


def to_ticket_evidence(ticket: RetrievedTicket) -> TicketEvidence:
    return TicketEvidence(
        ticket_id=ticket.ticket_id, product=ticket.product, issue=ticket.issue,
        resolution=ticket.resolution, similarity_score=ticket.similarity_score,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Validate config once; dependency outages are reported through readiness."""
    settings = settings or get_settings()
    application = FastAPI(
        title="Support Ticket RAG API", version="1.0.0",
        description="Grounded support resolutions using historical tickets and local Ollama.",
    )
    application.state.settings = settings

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        # Pydantic's default input/context can echo support text. Even an unknown
        # JSON key can contain sensitive text, so locations use known fields only.
        details = []
        for item in error.errors():
            location = [
                part if isinstance(part, int) or part in {"body", "issue", "top_k"} else "field"
                for part in item["loc"]
            ]
            details.append({"loc": location, "type": item["type"], "msg": "Invalid request value."})
        return JSONResponse(status_code=422, content={"detail": details})

    @application.exception_handler(ServiceError)
    async def unavailable(request: Request, error: ServiceError):
        LOGGER.warning("resolution_unavailable", extra={
            "event": "resolution_unavailable", "request_id": request.state.request_id,
            "reason": error.code,
        })
        return JSONResponse(status_code=503, content={
            "detail": "Resolution service is temporarily unavailable. Check the server logs."
        })

    @application.middleware("http")
    async def safe_errors(request: Request, call_next) -> Response:
        # Inside CORS so even unexpected failures have browser-visible headers.
        try:
            return await call_next(request)
        except Exception:
            LOGGER.error("request_failed", extra={
                "event": "request_failed", "request_id": request.state.request_id,
                "reason": "internal_error",
            })
            return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    application.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_allowed_origins),
        allow_credentials=False, allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"], expose_headers=["X-Request-ID"],
    )

    # Registered last, therefore outermost: also observes CORS preflights.
    @application.middleware("http")
    async def log_request(request: Request, call_next) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", supplied) else uuid4().hex
        request.state.request_id = request_id
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            LOGGER.error("request_failed", extra={
                "event": "request_failed", "request_id": request_id, "reason": "internal_error",
            })
            response = JSONResponse(status_code=500, content={"detail": "Internal server error."})
        response.headers["X-Request-ID"] = request_id
        # Route templates avoid logging arbitrary URL paths or query strings.
        route = request.scope.get("route")
        LOGGER.info("request_completed", extra={
            "event": "request_completed", "request_id": request_id,
            "method": request.method, "path": getattr(route, "path", "unmatched"),
            "status_code": response.status_code,
            "duration_ms": round((time.perf_counter() - started_at) * 1_000, 1),
        })
        return response

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    def health_check() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get(
        "/health/ready", response_model=ReadinessResponse,
        response_model_exclude_none=True, tags=["operations"],
        responses={503: {"model": ReadinessResponse, "description": "Dependencies unavailable"}},
    )
    def readiness_check(response: Response):
        # Sync endpoint runs blocking storage/network checks in FastAPI's thread pool.
        result = check_readiness(settings)
        response.status_code = 200 if result["status"] == "ready" else 503
        return result

    @application.post(
        "/v1/resolutions", response_model=ResolutionResponse, tags=["resolutions"],
        responses={503: {"description": "Resolution service unavailable"}},
    )
    def create_resolution(request: ResolutionRequest) -> ResolutionResponse:
        answer, tickets = generate_resolution(
            request.issue, top_k=request.top_k, model=settings.ollama_model, settings=settings,
        )
        return ResolutionResponse(
            answer=answer, evidence_sufficient=has_sufficient_evidence(tickets),
            evidence=[to_ticket_evidence(ticket) for ticket in tickets],
        )

    return application


app = create_app()
