
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
ARQUIVO = BASE / "data" / "raw" / "mortalidade_infantil.csv"

df = pd.read_csv(ARQUIVO, low_memory=False)

print("\nCOLUNAS DO ARQUIVO ORIGINAL")
print(df.columns.tolist())

paises = [
    "Central African Republic",
    "Somalia",
    "South Sudan",
    "Malawi"
]

print("\nREGISTROS ORIGINAIS â€” PAÃSES INVESTIGADOS")

filtro = (
    df["Entity"].isin(paises)
    & df["Year"].between(2008, 2024)
)

colunas = [
    c for c in df.columns
    if c in ["Entity", "Code", "Year"]
    or "mortality" in c.lower()
    or "source" in c.lower()
]

print(
    df.loc[filtro, colunas]
    .sort_values(["Entity", "Year"])
    .to_string(index=False)
)

print("\nREGISTROS DUPLICADOS")
duplicados = df.duplicated(
    subset=["Entity", "Code", "Year"],
    keep=False
)

print(df.loc[duplicados, colunas].to_string(index=False))

