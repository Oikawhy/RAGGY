def test_production_health_endpoint_is_defined():
    from app.main import create_production_app
    app = create_production_app()
    paths = {r.path for r in app.routes}
    assert "/health" in paths
    assert "/ask" in paths


def test_production_app_has_lifespan():
    from app.main import create_production_app
    app = create_production_app()
    assert app.router.lifespan_context is not None
