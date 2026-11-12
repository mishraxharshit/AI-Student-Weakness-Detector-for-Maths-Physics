"""Three ways to score how weak a student is in a topic. All return a DataFrame
[student_id, topic, score] where HIGHER score = STRONGER (so weakness = -score).

1. RawAccuracy      - fraction correct (the naive spreadsheet approach).
2. BetaShrinkage    - empirical-Bayes posterior mean accuracy; small samples are pulled toward the topic average.
3. HierarchicalIRT  - regularised logistic regression with student, student x topic and question effects:
                      logit P(correct) = s[student] + st[student,topic] - d[question]
                      It separates *question difficulty* from *student ability* and shares information
                      across a student's topics (partial pooling), and can predict answers to unseen questions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression


class RawAccuracy:
    name = "Raw accuracy"

    def fit(self, df: pd.DataFrame):
        self.table = df.groupby(["student_id", "topic"], as_index=False).correct.mean().rename(columns={"correct": "score"})
        return self

    def scores(self) -> pd.DataFrame:
        return self.table


class BetaShrinkage:
    name = "Beta shrinkage"

    def __init__(self, prior_strength: float = 4.0):
        self.k = prior_strength

    def fit(self, df: pd.DataFrame):
        topic_mean = df.groupby("topic").correct.mean()
        g = df.groupby(["student_id", "topic"]).correct.agg(["sum", "count"]).reset_index()
        prior = g.topic.map(topic_mean)
        g["score"] = (g["sum"] + self.k * prior) / (g["count"] + self.k)
        self.table = g[["student_id", "topic", "score"]]
        return self

    def scores(self) -> pd.DataFrame:
        return self.table


class HierarchicalIRT:
    name = "Hierarchical IRT"

    def __init__(self, C_student: float = 1.0, C_pair: float = 0.5, C_question: float = 1.0):
        # per-block L2 strengths are implemented by rescaling columns (ridge with different priors)
        self.scale = dict(student=np.sqrt(C_student), pair=np.sqrt(C_pair), question=np.sqrt(C_question))

    def _design(self, df: pd.DataFrame, fit: bool):
        if fit:
            self.students = {s: i for i, s in enumerate(sorted(df.student_id.unique()))}
            self.pairs = {p: i for i, p in enumerate(sorted(set(zip(df.student_id, df.topic))))}
            self.questions = {q: i for i, q in enumerate(sorted(df.question_id.unique()))}
        n = len(df)
        r = np.arange(n)
        si = df.student_id.map(self.students).to_numpy()
        pi = np.array([self.pairs.get(p, -1) for p in zip(df.student_id, df.topic)])
        qi = df.question_id.map(self.questions).fillna(-1).astype(int).to_numpy()
        S = sparse.csr_matrix((np.where(si >= 0, self.scale["student"], 0.0), (r, np.maximum(si, 0))), shape=(n, len(self.students)))
        P = sparse.csr_matrix((np.where(pi >= 0, self.scale["pair"], 0.0), (r, np.maximum(pi, 0))), shape=(n, len(self.pairs)))
        Q = sparse.csr_matrix((np.where(qi >= 0, -self.scale["question"], 0.0), (r, np.maximum(qi, 0))), shape=(n, len(self.questions)))
        return sparse.hstack([S, P, Q]).tocsr()

    def fit(self, df: pd.DataFrame):
        X = self._design(df, fit=True)
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, df.correct.to_numpy())
        w = self.clf.coef_.ravel()
        ns, npair = len(self.students), len(self.pairs)
        self.w_student = w[:ns] * self.scale["student"]
        self.w_pair = w[ns:ns + npair] * self.scale["pair"]
        self.w_question = w[ns + npair:] * self.scale["question"]   # = difficulty (higher = harder)
        return self

    def scores(self) -> pd.DataFrame:
        rows = [(s, t, self.w_student[self.students[s]] + self.w_pair[i]) for (s, t), i in self.pairs.items()]
        return pd.DataFrame(rows, columns=["student_id", "topic", "score"])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._design(df, fit=False))[:, 1]

    def difficulty(self) -> pd.Series:
        return pd.Series(self.w_question, index=list(self.questions))
