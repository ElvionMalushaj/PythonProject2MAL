"""
eda.py
------
Explorative Datenanalyse (EDA) und deskriptive Statistik fuer den WDBC
Datensatz. Erzeugt die Abbildungen, die im Bericht verwendet werden, und
speichert sie in outputs/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import FEATURE_COLUMNS, PROJECT_ROOT

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Einheitliches, ruhiges Erscheinungsbild fuer alle Plots.
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = {"B (gutartig)": "#4C72B0", "M (boesartig)": "#C44E52"}


def _diagnosis_label(df: pd.DataFrame) -> pd.Series:
    """Lesbare Klassenbezeichnung fuer Legenden."""
    return df["diagnosis"].map({"B": "B (gutartig)", "M": "M (boesartig)"})


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Deskriptive Statistik-Tabelle der 30 Merkmale (transponiert)."""
    desc = df[FEATURE_COLUMNS].describe().T
    desc = desc[["mean", "std", "min", "50%", "max"]]
    desc = desc.rename(columns={"50%": "median"})
    return desc.round(3)


def plot_class_distribution(df: pd.DataFrame) -> Path:
    """Balkendiagramm der Klassenverteilung (B vs. M)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["diagnosis"].value_counts().reindex(["B", "M"])
    labels = ["B (gutartig)", "M (boesartig)"]
    bars = ax.bar(labels, counts.values, color=[PALETTE[l] for l in labels])
    for bar, value in zip(bars, counts.values):
        pct = 100 * value / len(df)
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3,
                f"{value}\n({pct:.1f} %)", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Anzahl Faelle")
    ax.set_title("Klassenverteilung im WDBC-Datensatz (n = 569)")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    path = FIGURE_DIR / "01_class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_distributions(df: pd.DataFrame) -> Path:
    """Histogramme/KDE ausgewaehlter Merkmale, getrennt nach Diagnose.

    Zeigt, dass sich die Klassen bei mehreren Merkmalen deutlich trennen.
    """
    features = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "concavity_mean", "concave_points_worst",
    ]
    df_plot = df.copy()
    df_plot["Diagnose"] = _diagnosis_label(df_plot)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, feature in zip(axes.flat, features):
        for label, color in PALETTE.items():
            subset = df_plot.loc[df_plot["Diagnose"] == label, feature]
            sns.kdeplot(subset, ax=ax, fill=True, alpha=0.4,
                        color=color, label=label, linewidth=1.5)
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("Dichte")
    axes.flat[0].legend(title="Diagnose", fontsize=8)
    fig.suptitle("Verteilung ausgewaehlter Merkmale nach Diagnose", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = FIGURE_DIR / "02_feature_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_boxplots(df: pd.DataFrame) -> Path:
    """Boxplots einiger Merkmale nach Diagnose (Ausreisser / Streuung)."""
    features = ["radius_mean", "texture_mean", "area_mean", "concavity_mean"]
    df_plot = df.copy()
    df_plot["Diagnose"] = _diagnosis_label(df_plot)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    for ax, feature in zip(axes, features):
        sns.boxplot(data=df_plot, x="Diagnose", y=feature, ax=ax,
                    hue="Diagnose", palette=PALETTE, legend=False)
        ax.set_title(feature)
        ax.set_xlabel("")
    fig.suptitle("Boxplots ausgewaehlter Merkmale nach Diagnose", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIGURE_DIR / "03_boxplots.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Korrelationsmatrix aller 30 Merkmale als Heatmap."""
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.7, "label": "Pearson-Korrelation"},
                xticklabels=True, yticklabels=True, ax=ax)
    ax.set_title("Korrelationsmatrix der 30 Merkmale", fontsize=14)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    path = FIGURE_DIR / "04_correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def top_correlations_with_target(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """Die n Merkmale mit der staerksten Korrelation zur Zielvariable."""
    corr = df[FEATURE_COLUMNS + ["target"]].corr()["target"].drop("target")
    return corr.abs().sort_values(ascending=False).head(n)


def plot_target_correlations(df: pd.DataFrame) -> Path:
    """Balkendiagramm: staerkste Korrelationen mit der Diagnose."""
    corr = df[FEATURE_COLUMNS + ["target"]].corr()["target"].drop("target")
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_xlabel("Pearson-Korrelation mit target (M = 1)")
    ax.set_title("Staerkste Zusammenhaenge mit der Diagnose")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    path = FIGURE_DIR / "05_target_correlations.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pca_scatter(df: pd.DataFrame) -> Path:
    """2D-PCA-Streudiagramm zur Visualisierung der Klassentrennbarkeit."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = df[FEATURE_COLUMNS].values
    X_scaled = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2, random_state=42).fit(X_scaled)
    coords = pcs.transform(X_scaled)

    df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"])
    df_plot["Diagnose"] = _diagnosis_label(df).values

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color in PALETTE.items():
        subset = df_plot[df_plot["Diagnose"] == label]
        ax.scatter(subset["PC1"], subset["PC2"], s=18, alpha=0.7,
                   color=color, label=label, edgecolor="white", linewidth=0.3)
    var = pcs.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({var[0]:.1f} % Varianz)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f} % Varianz)")
    ax.set_title("PCA-Projektion des WDBC-Datensatzes")
    ax.legend(title="Diagnose")
    fig.tight_layout()
    path = FIGURE_DIR / "06_pca_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_full_eda(df: pd.DataFrame) -> dict:
    """Erzeugt alle EDA-Abbildungen und gibt ein Ergebnis-Dictionary zurueck."""
    print("Erzeuge EDA-Abbildungen ...")
    paths = {
        "class_distribution": plot_class_distribution(df),
        "feature_distributions": plot_feature_distributions(df),
        "boxplots": plot_boxplots(df),
        "correlation_heatmap": plot_correlation_heatmap(df),
        "target_correlations": plot_target_correlations(df),
        "pca_scatter": plot_pca_scatter(df),
    }
    for name, path in paths.items():
        print(f"  gespeichert: {path.relative_to(PROJECT_ROOT)}")
    return {
        "figures": paths,
        "describe": describe_dataset(df),
        "top_correlations": top_correlations_with_target(df),
    }


if __name__ == "__main__":
    from data_loader import load_data

    data = load_data()
    result = run_full_eda(data)
    print("\nDeskriptive Statistik (Auszug):")
    print(result["describe"].head())
    print("\nStaerkste Korrelationen mit der Diagnose:")
    print(result["top_correlations"])
