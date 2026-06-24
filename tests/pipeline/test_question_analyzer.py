from app.pipeline.question_analyzer import analyze_question


def test_detects_calculation_question():
    analysis = analyze_question("Порахуй індексацію для працівника із зарплатою 25000 грн")
    assert analysis.requires_calculation is True


def test_non_calculation_policy_question_is_not_calculation():
    analysis = analyze_question("Чи можна оплатити лікарняний без медичного документа?")
    assert analysis.requires_calculation is False
