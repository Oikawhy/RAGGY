from scripts.run_evaluation import load_questions


def test_load_questions_reads_assignment_questions():
    questions = load_questions("data/test_questions.json")
    assert len(questions) == 6
    assert questions[0]["id"] == "q001"
    assert "question" in questions[0]
