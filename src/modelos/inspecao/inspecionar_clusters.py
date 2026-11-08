
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

arquivo = BASE / "data" / "processed" / "painel_clusters.csv"

if not arquivo.exists():
    print("Arquivo painel_clusters.csv não encontrado.")
    raise SystemExit

df = pd.read_csv(arquivo)

print("=" * 70)
print("INSPEÇÃO DO ARQUIVO DE CLUSTERS")
print("=" * 70)

print("\nDimensões:")
print(df.shape)

print("\nColunas:")
print(df.columns.tolist())

print("\nTipos:")
print(df.dtypes)

print("\nPrimeiros registros:")
print(df.head(10).to_string(index=False))

print("\nQuantidade de perfis:")
if "perfil" in df.columns:
    print(df["perfil"].value_counts(dropna=False))

print("\nValores ausentes:")
print(df.isna().sum())

print("\nPeríodo:")
if "Year" in df.columns:
    print(df["Year"].min(), "-", df["Year"].max())

print("\nINSPEÇÃO CONCLUÍDA.")
