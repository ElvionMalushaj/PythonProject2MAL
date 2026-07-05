"""
data_loader.py
--------------
Laden und Grundaufbereitung des Wisconsin Diagnostic Breast Cancer (WDBC)
Datensatzes.

Der Datensatz besteht aus 569 Fine-Needle-Aspirate (FNA) Aufnahmen von
Brustgewebe. Fuer jeden Zellkern werden 10 Basismerkmale berechnet
(radius, texture, perimeter, ...). Von jedem dieser Merkmale liegen der
Mittelwert (_mean), der Standardfehler (_se) und der "worst"-Wert
(Mittelwert der drei groessten Werte, _worst) vor -> 30 Merkmale.

Zielvariable: diagnosis (M = malignant/boesartig, B = benign/gutartig).
"""

from pathlib import Path

import pandas as pd

# Die 10 Basismerkmale in der Reihenfolge, in der sie in wdbc.data auftauchen.
BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

# Vollstaendige Spaltennamen: id, diagnosis, danach _mean, _se, _worst.
COLUMN_NAMES = (
    ["id", "diagnosis"]
    + [f"{f}_mean" for f in BASE_FEATURES]
    + [f"{f}_se" for f in BASE_FEATURES]
    + [f"{f}_worst" for f in BASE_FEATURES]
)

# Alle 30 Merkmalsspalten (ohne id/diagnosis).
FEATURE_COLUMNS = COLUMN_NAMES[2:]

# Projekt-Wurzelverzeichnis (eine Ebene ueber src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wdbc.data"


def load_data(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Liest wdbc.data ein und gibt ein aufbereitetes DataFrame zurueck.

    - Setzt sinnvolle Spaltennamen.
    - Entfernt die nicht informative id-Spalte.
    - Legt eine binaere Zielspalte ``target`` an (M=1, B=0).
    """
    path = Path(path)
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)

    # id ist ein reiner Identifikator und traegt keine Vorhersageinformation.
    df = df.drop(columns=["id"])

    # Binaere Kodierung: boesartig (M) = 1 ist die "positive" Klasse,
    # denn das ist der medizinisch relevante Fall, den wir erkennen wollen.
    df["target"] = (df["diagnosis"] == "M").astype(int)

    return df


def get_feature_matrix(df: pd.DataFrame):
    """Trennt Merkmalsmatrix X (30 Spalten) und Zielvektor y."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].copy()
    return X, y


if __name__ == "__main__":
    data = load_data()
    print(f"Datensatz geladen: {data.shape[0]} Zeilen, {data.shape[1]} Spalten")
    print(data["diagnosis"].value_counts())
    print(data.head())
