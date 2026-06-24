from app.pipeline.orchestrator import PipelineOrchestrator
from app.models import AskRequest


def test_orchestrator_returns_fallback_when_calculation_context_is_insufficient(fake_pipeline_deps):
    orchestrator = PipelineOrchestrator(**fake_pipeline_deps.with_no_formula_context())
    response = orchestrator.ask(AskRequest(question="Порахуй індексацію для зарплати 25000 грн"))
    assert response.confidence == "low"
    assert response.fallback_reason is not None
    assert "недостатньо" in response.answer.lower()


def test_orchestrator_returns_sources_on_success(fake_pipeline_deps):
    orchestrator = PipelineOrchestrator(**fake_pipeline_deps.with_vacation_context())
    response = orchestrator.ask(AskRequest(question="Чи можна у відпустку після 3 місяців?"))
    assert response.sources
    assert response.trace_id
