"""Shared user-input rules; presentation belongs to each entry point."""

MIN_ISSUE_LENGTH = 5
MAX_ISSUE_LENGTH = 2000
MIN_TOP_K = 1
MAX_TOP_K = 5


class InputValidationError(ValueError):
    """A user-correctable error that never echoes the supplied input."""


def validate_issue(value: str) -> str:
    if not isinstance(value, str):
        raise InputValidationError("Issue must be text.")
    value = value.strip()
    if not MIN_ISSUE_LENGTH <= len(value) <= MAX_ISSUE_LENGTH:
        raise InputValidationError("Issue must contain between 5 and 2000 characters.")
    return value


def validate_top_k(value: int) -> int:
    if type(value) is not int or not MIN_TOP_K <= value <= MAX_TOP_K:
        raise InputValidationError("top_k must be an integer between 1 and 5.")
    return value


def validate_model(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise InputValidationError("Model must be nonblank text without control characters.")
    return value.strip()
