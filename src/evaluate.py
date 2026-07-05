"""
evaluate.py
-----------
Bewertung der trainierten Modelle auf dem Testset und Erzeugung der
Ergebnis-Abbildungen (Konfusionsmatrix, ROC-Kurve, Feature-Importance,
CV-Vergleich). Der F1-Score steht dabei im Mittelpunkt, da bei der
Krebsdiagnose sowohl falsch-negative als auch falsch-positive Vorhersagen
teuer sind und der F1-Score Precision und Recall ausbalanciert.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data_loader import PROJECT_ROOT
from model import TrainedModel, get_feature_importances

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


def evaluate_on_test(model: TrainedModel, X_test, y_test) -> dict:
    """Berechnet die zentralen Kennzahlen auf dem Testset."""
    y_pred = model.estimator.predict(X_test)
    y_proba = model.estimator.predict_proba(X_test)[:, 1]
    return {
        "name": model.name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred, target_names=["B (gutartig)", "M (boesartig)"]
        ),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def plot_confusion_matrix(result: dict, y_test) -> Path:
    """Konfusionsmatrix des besten Modells."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=result["confusion_matrix"],
        display_labels=["B (gutartig)", "M (boesartig)"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Konfusionsmatrix – {result['name']}")
    ax.set_xlabel("Vorhergesagte Klasse")
    ax.set_ylabel("Tatsaechliche Klasse")
    fig.tight_layout()
    path = FIGURE_DIR / "07_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curves(results: "list[dict]", y_test) -> Path:
    """ROC-Kurven aller Modelle im Vergleich."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{res['name']} (AUC = {res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Zufall")
    ax.set_xlabel("Falsch-Positiv-Rate")
    ax.set_ylabel("Richtig-Positiv-Rate (Recall)")
    ax.set_title("ROC-Kurven im Vergleich")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = FIGURE_DIR / "08_roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(model: TrainedModel, feature_names, n: int = 15) -> Path:
    """Balkendiagramm der wichtigsten Merkmale (Random Forest)."""
    importances = get_feature_importances(model, feature_names)[:n]
    names = [p[0] for p in importances][::-1]
    values = [p[1] for p in importances][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, values, color="#55A868")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {n} Merkmale – {model.name}")
    fig.tight_layout()
    path = FIGURE_DIR / "09_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cv_comparison(models: "list[TrainedModel]") -> Path:
    """Vergleich der Cross-Validation-F1-Scores (Mittelwert +/- Std)."""
    names = [m.name for m in models]
    means = [m.cv_f1_mean for m in models]
    stds = [m.cv_f1_std for m in models]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=["#4C72B0", "#55A868"])
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("F1-Score (5-fache CV)")
    ax.set_title("Cross-Validation-Vergleich (F1)")
    ax.set_ylim(0.9, 1.0)
    plt.setp(ax.get_xticklabels(), rotation=8, ha="right")
    fig.tight_layout()
    path = FIGURE_DIR / "10_cv_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
