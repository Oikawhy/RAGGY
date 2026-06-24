import pytest

from app.main import create_app


def test_ask_route_registered():
    app = create_app({})
    paths_and_methods = {(r.path, tuple(r.methods)) for r in app.routes if hasattr(r, "methods")}
    assert ("/ask", ("POST",)) in paths_and_methods


def test_health_route_registered():
    app = create_app({})
    paths_and_methods = {(r.path, tuple(r.methods)) for r in app.routes if hasattr(r, "methods")}
    assert ("/health", ("GET",)) in paths_and_methods
