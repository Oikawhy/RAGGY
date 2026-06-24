from scripts.ingest_kb import build_ingestion_plan


def test_ingestion_plan_counts_sections_and_chunks():
    plan = build_ingestion_plan("data/knowledge_base.md", logical_source_key="assignment-kb", version=1)
    assert plan.source.logical_source_key == "assignment-kb"
    assert plan.total_sections == 10
    assert plan.total_chunks > 10
    assert len(plan.corpus_hash) == 16
