"""RFC 7807 problem+json exception handler for DRF."""
from __future__ import annotations

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler


class DomainError(Exception):
    """Raised by service/use-case code for user-facing business errors."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    type_: str = "about:blank"

    def __init__(self, detail: str, *, status_code: int | None = None, type_: str | None = None):
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code
        if type_ is not None:
            self.type_ = type_
        super().__init__(detail)


class RiskRejected(DomainError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    type_ = "https://alphadesk.ai/errors/risk-rejected"


class TenantMismatch(DomainError):
    status_code = status.HTTP_403_FORBIDDEN
    type_ = "https://alphadesk.ai/errors/tenant-mismatch"


def problem_json_handler(exc, context):
    response = exception_handler(exc, context)
    if isinstance(exc, DomainError):
        return Response(
            {
                "type": exc.type_,
                "title": exc.__class__.__name__,
                "status": exc.status_code,
                "detail": exc.detail,
            },
            status=exc.status_code,
            content_type="application/problem+json",
        )
    if response is not None:
        response.data = {
            "type": "about:blank",
            "title": exc.__class__.__name__,
            "status": response.status_code,
            "detail": response.data if isinstance(response.data, str) else response.data,
        }
        response["Content-Type"] = "application/problem+json"
    return response
