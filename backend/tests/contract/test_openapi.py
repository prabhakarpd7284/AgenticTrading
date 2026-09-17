"""OpenAPI contract tests — make sure drf-spectacular keeps emitting a
schema that matches the actual API behaviour.

The heavy lifting is delegated to schemathesis, which does property-based
API fuzzing from the schema. This file runs fast smoke checks only; full
fuzzing lives in CI behind a dedicated job (see docs/TESTING_STRATEGY.md §6).
"""
from __future__ import annotations

import pytest
from django.urls import reverse


pytestmark = pytest.mark.django_db


def test_openapi_schema_endpoint_reachable(auth_client):
    resp = auth_client.get("/api/schema/")
    assert resp.status_code == 200
    assert "openapi" in resp.content.decode()[:50].lower()


def test_openapi_schema_contains_core_paths(auth_client):
    """Smoke test — the money-critical endpoints must appear in the schema.
    If a new path ships without a schema entry, customers can't integrate."""
    resp = auth_client.get("/api/schema/")
    body = resp.content.decode()
    for path in ["/orders/", "/portfolios/", "/agents/runs/"]:
        assert path in body, f"missing {path} in OpenAPI schema"
