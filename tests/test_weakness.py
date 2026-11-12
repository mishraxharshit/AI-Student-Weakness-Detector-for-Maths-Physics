import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from weakness import simulate as sim  # noqa: E402
from weakness.cli import main as cli_main  # noqa: E402
from weakness.estimators import BetaShrinkage, HierarchicalIRT, RawAccuracy  # noqa: E402
from weakness.recommend import Advisor  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return sim.simulate(n_students=60, seed=1, attempts_per_topic=8)


def test_simulation_schema_and_reproducibility(data):
    attempts, truth, qdiff = data
    assert set(attempts.columns) == {"student_id", "topic", "question_id", "attempt_order", "correct"}
    assert set(attempts.correct) <= {0, 1}
    assert len(attempts) == 60 * len(sim.TOPICS) * 8
    # no student answers the same question twice
    assert not attempts.duplicated(["student_id", "question_id"]).any()
    again = sim.simulate(n_students=60, seed=1, attempts_per_topic=8)[0]
    pd.testing.assert_frame_equal(attempts, again)


def test_stronger_students_answer_more_correctly(data):
    attempts, truth, _ = data
    acc = attempts.groupby(["student_id", "topic"]).correct.mean().reset_index().merge(truth, on=["student_id", "topic"])
    assert np.corrcoef(acc.correct, acc.true_ability)[0, 1] > 0.4


def test_raw_accuracy_is_exact_fraction():
    df = pd.DataFrame({"student_id": ["a"] * 4, "topic": ["Algebra"] * 4, "question_id": list("wxyz"), "correct": [1, 1, 0, 0]})
    assert RawAccuracy().fit(df).scores().score.iloc[0] == 0.5


def test_shrinkage_pulls_small_samples_toward_topic_mean():
    df = pd.DataFrame({"student_id": ["a", "b", "b", "b", "b"], "topic": ["T"] * 5,
                       "question_id": list("12345"), "correct": [1, 0, 1, 0, 1]})
    s = BetaShrinkage(prior_strength=4).fit(df).scores().set_index("student_id").score
    assert 0.6 < s["a"] < 1.0 and s["a"] < 1.0       # one correct answer is NOT 100 %
    assert s["a"] == pytest.approx((1 + 4 * 0.6) / 5)


def test_irt_recovers_difficulty_ordering(data):
    attempts, _, qdiff = data
    m = HierarchicalIRT().fit(attempts)
    est = m.difficulty().rename("est").reset_index().rename(columns={"index": "question_id"})
    j = est.merge(qdiff, on="question_id")
    assert np.corrcoef(j.est, j.true_difficulty)[0, 1] > 0.7


def test_irt_predictions_are_probabilities(data):
    attempts, _, _ = data
    p = HierarchicalIRT().fit(attempts).predict_proba(attempts.head(50))
    assert ((p > 0) & (p < 1)).all()


def test_advisor_recommends_only_unseen_questions_in_weak_topics(data):
    attempts, _, _ = data
    adv = Advisor(attempts)
    sid = attempts.student_id.iloc[0]
    weak = adv.weakest_topics(sid, 2)
    recs = adv.recommend_questions(sid, n_per_topic=3, k_topics=2)
    seen = set(attempts[attempts.student_id == sid].question_id)
    assert set(recs.topic) <= set(weak.topic)
    assert not (set(recs.question_id) & seen)
    assert recs.predicted_success.between(0, 1).all()


def test_cli_runs_and_rejects_bad_input(tmp_path, capsys):
    attempts, _, _ = sim.simulate(n_students=20, seed=0, attempts_per_topic=5)
    f = tmp_path / "q.csv"; attempts.to_csv(f, index=False)
    cli_main(["--data", str(f), "--student", "S003"])
    assert "Weakest topics for S003" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli_main(["--data", str(f), "--student", "NOPE"])
    bad = tmp_path / "bad.csv"; pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    with pytest.raises(SystemExit):
        cli_main(["--data", str(bad), "--student", "S003"])
