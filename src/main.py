"""
main.py
-------
Vollstaendige Analyse-Pipeline fuer das WDBC-Projekt.

Ablauf:
  1. Daten laden und aufbereiten.
  2. Explorative Datenanalyse (Abbildungen erzeugen).
  3. Train/Test-Split (stratifiziert).
  4. Baseline (Logistic Regression) und Hauptmodell (Random Forest, getunt)
     trainieren und per Cross-Validation bewerten.
  5. Beide Modelle auf dem Testset evaluieren (Fokus: F1-Score).
  6. Ergebnis-Abbildungen und eine metrics.json speichern.

Aufruf aus dem Projektverzeichnis:
    python src/main.py
"""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from data_loader import FEATURE_COLUMNS, PROJECT_ROOT, get_feature_matrix, load_data
from eda import run_full_eda
from evaluate import (
    evaluate_on_test,
    plot_confusion_matrix,
    plot_cv_comparison,
    plot_feature_importance,
    plot_roc_curves,
)
from model import get_feature_importances, train_baseline, tune_random_forest

RANDOM_STATE = 42
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _print_header(text: str) -> None:
    print("\n" + "=" * 62)
    print(text)
    print("=" * 62)


def main() -> dict:
    _print_header("1) Daten laden")
    df = load_data()
    print(f"{df.shape[0]} Faelle, {len(FEATURE_COLUMNS)} Merkmale")
    print(df["diagnosis"].value_counts().to_string())

    _print_header("2) Explorative Datenanalyse")
    eda_result = run_full_eda(df)

    _print_header("3) Train/Test-Split (80/20, stratifiziert)")
    X, y = get_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Training: {len(X_train)} | Test: {len(X_test)}")

    _print_header("4) Modelle trainieren")
    print("Trainiere Baseline (Logistic Regression) ...")
    baseline = train_baseline(X_train, y_train)
    print(f"  CV-F1: {baseline.cv_f1_mean:.4f} (+/- {baseline.cv_f1_std:.4f})")

    print("Tuning Random Forest (GridSearchCV) ...")
    rf = tune_random_forest(X_train, y_train)
    print(f"  Beste Parameter: {rf.best_params}")
    print(f"  CV-F1: {rf.cv_f1_mean:.4f} (+/- {rf.cv_f1_std:.4f})")

    models = [baseline, rf]

    _print_header("5) Evaluation auf dem Testset")
    results = [evaluate_on_test(m, X_test, y_test) for m in models]
    for res in results:
        print(f"\n--- {res['name']} ---")
        print(f"Accuracy : {res['accuracy']:.4f}")
        print(f"Precision: {res['precision']:.4f}")
        print(f"Recall   : {res['recall']:.4f}")
        print(f"F1-Score : {res['f1']:.4f}")
        print(f"ROC-AUC  : {res['roc_auc']:.4f}")

    # Bestes Modell nach F1 auf dem Testset.
    best_result = max(results, key=lambda r: r["f1"])
    best_model = models[results.index(best_result)]
    print(f"\nBestes Modell (F1): {best_result['name']}")
    print("\nClassification Report:")
    print(best_result["report"])

    _print_header("6) Ergebnis-Abbildungen speichern")
    fig_paths = {
        "confusion_matrix": plot_confusion_matrix(best_result, y_test),
        "roc_curves": plot_roc_curves(results, y_test),
        "feature_importance": plot_feature_importance(rf, FEATURE_COLUMNS),
        "cv_comparison": plot_cv_comparison(models),
    }
    for name, path in fig_paths.items():
        print(f"  gespeichert: {path.relative_to(PROJECT_ROOT)}")

    # Ergebnisse fuer den Bericht als JSON ablegen.
    top_importances = get_feature_importances(rf, FEATURE_COLUMNS)[:15]
    summary = {
        "dataset": {
            "n_samples": int(df.shape[0]),
            "n_features": len(FEATURE_COLUMNS),
            "n_benign": int((df["target"] == 0).sum()),
            "n_malignant": int((df["target"] == 1).sum()),
        },
        "split": {"n_train": int(len(X_train)), "n_test": int(len(X_test))},
        "best_params_random_forest": rf.best_params,
        "cross_validation_f1": {m.name: {
            "mean": m.cv_f1_mean, "std": m.cv_f1_std,
            "folds": [round(float(s), 4) for s in m.cv_f1_scores],
        } for m in models},
        "test_metrics": {res["name"]: {
            "accuracy": round(res["accuracy"], 4),
            "precision": round(res["precision"], 4),
            "recall": round(res["recall"], 4),
            "f1": round(res["f1"], 4),
            "roc_auc": round(res["roc_auc"], 4),
            "confusion_matrix": res["confusion_matrix"].tolist(),
        } for res in results},
        "best_model": best_result["name"],
        "top_feature_importances": [
            {"feature": f, "importance": round(float(v), 4)} for f, v in top_importances
        ],
        "top_target_correlations": {
            k: round(float(v), 4) for k, v in eda_result["top_correlations"].items()
        },
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  gespeichert: {metrics_path.relative_to(PROJECT_ROOT)}")

    # Deskriptive Statistik zusaetzlich als CSV.
    desc_path = OUTPUT_DIR / "descriptive_statistics.csv"
    eda_result["describe"].to_csv(desc_path)
    print(f"  gespeichert: {desc_path.relative_to(PROJECT_ROOT)}")

    _print_header("Fertig")
    print("Alle Abbildungen liegen in outputs/figures/.")
    print("Kennzahlen in outputs/metrics.json.")
    return summary


if __name__ == "__main__":
    main()
