from crag.evaluator import Action, decide_action


def test_any_confident_doc_is_correct():
    assert decide_action([1, 8, 2], upper=7, lower=3) is Action.CORRECT


def test_all_low_is_incorrect():
    assert decide_action([0, 3, 1], upper=7, lower=3) is Action.INCORRECT


def test_in_between_is_ambiguous():
    assert decide_action([2, 5, 1], upper=7, lower=3) is Action.AMBIGUOUS
