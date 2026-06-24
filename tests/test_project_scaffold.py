from pathlib import Path


def test_required_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app" / "__init__.py").exists()
    assert (root / "data" / "knowledge_base.md").exists()
    assert (root / "data" / "test_questions.json").exists()
    assert (root / ".env.example").exists()
    assert (root / "requirements.txt").exists()
