"""
model.py
--------
Trainings-Pipeline fuer die Diagnose-Vorhersage.

Hauptmodell: Random Forest (tiefe Analyse mit Cross-Validation,
Hyperparameter-Tuning und Feature-Importance).
Zum Vergleich wird eine einfache Logistic-Regression-Baseline trainiert.

Alle Modelle werden in einer sklearn-Pipeline mit StandardScaler
verpackt, damit die Skalierung ausschliesslich auf den Trainingsdaten
angepasst wird (kein Data Leakage).
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


@dataclass
class TrainedModel:
    """Buendelt ein trainiertes Modell mit seinen Metadaten."""
    name: str
    estimator: Pipeline
    cv_f1_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    best_params: dict = field(default_factory=dict)

    @property
    def cv_f1_mean(self) -> float:
        return float(self.cv_f1_scores.mean()) if self.cv_f1_scores.size else float("nan")

    @property
    def cv_f1_std(self) -> float:
        return float(self.cv_f1_scores.std()) if self.cv_f1_scores.size else float("nan")


def _cv() -> StratifiedKFold:
    """Stratifizierte 5-fache Kreuzvalidierung (erhaelt Klassenverhaeltnis)."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def build_baseline() -> Pipeline:
    """Logistic-Regression-Baseline mit Standardisierung."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])


def build_random_forest() -> Pipeline:
    """Random-Forest-Pipeline (Skalierung fuer RF nicht noetig, aber
    einheitlich zur Baseline; schadet dem RF nicht)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def tune_random_forest(X_train, y_train) -> TrainedModel:
    """Hyperparameter-Tuning des Random Forest per GridSearchCV (F1-optimiert)."""
    pipe = build_random_forest()
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
    }
    search = GridSearchCV(
        pipe, param_grid, scoring="f1", cv=_cv(), n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    cv_scores = cross_val_score(best, X_train, y_train, scoring="f1", cv=_cv())
    return TrainedModel(
        name="Random Forest (getunt)",
        estimator=best,
        cv_f1_scores=cv_scores,
        best_params={k.replace("clf__", ""): v for k, v in search.best_params_.items()},
    )


def train_baseline(X_train, y_train) -> TrainedModel:
    """Trainiert die Baseline und bewertet sie per Cross-Validation."""
    pipe = build_baseline()
    cv_scores = cross_val_score(pipe, X_train, y_train, scoring="f1", cv=_cv())
    pipe.fit(X_train, y_train)
    return TrainedModel(
        name="Logistic Regression (Baseline)",
        estimator=pipe,
        cv_f1_scores=cv_scores,
    )


def get_feature_importances(model: TrainedModel, feature_names) -> "list[tuple[str, float]]":
    """Feature-Importances des Random Forest, absteigend sortiert."""
    clf = model.estimator.named_steps["clf"]
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        return []
    pairs = list(zip(feature_names, importances))
    return sorted(pairs, key=lambda p: p[1], reverse=True)
