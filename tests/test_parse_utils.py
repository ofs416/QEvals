from parse_utils import valid_generation


def test_valid_generation_accepts_nonempty_questions():
    assert valid_generation({"questions": [{"text": "Find x", "marks": 2}]}) is True


def test_valid_generation_rejects_fragment_without_questions():
    # GLM chain-of-thought fragment: parses as JSON but is not a generation
    assert valid_generation({"tag": "M1", "text": "Uses correct formula"}) is False


def test_valid_generation_rejects_empty_questions_list():
    assert valid_generation({"questions": []}) is False


def test_valid_generation_rejects_non_list_questions():
    assert valid_generation({"questions": "Q1: find x"}) is False


def test_valid_generation_rejects_none():
    assert valid_generation(None) is False
