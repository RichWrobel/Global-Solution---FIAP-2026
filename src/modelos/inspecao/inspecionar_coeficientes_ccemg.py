
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

arquivo = (
    BASE
    / "data"
    / "processed"
    / "modelo_ccemg"
    / "coeficientes_por_pais.csv"
)

df = pd.read_csv(arquivo)

print("=" * 80)
print("ESTRUTURA DO ARQUIVO DE COEFICIENTES")
print("=" * 80)

print("\n1. DIMENSÕES")
print("Linhas:", len(df))
print("Colunas:", len(df.columns))

print("\n2. NOMES DAS COLUNAS")
print(df.columns.tolist())

print("\n3. PRIMEIRAS 15 LINHAS")
print(df.head(15).to_string(index=False))

print("\n4. TIPOS DE MODELO")
if "modelo" in df.columns:
    print(df["modelo"].value_counts(dropna=False).to_string())

print("\n5. LIMITES MÍNIMOS DE OBSERVAÇÕES")
if "minimo_observacoes" in df.columns:
    print(df["minimo_observacoes"].value_counts(dropna=False).to_string())
elif "min_obs" in df.columns:
    print(df["min_obs"].value_counts(dropna=False).to_string())
else:
    print("Não foi encontrada coluna com esse nome.")

print("\n6. PAÍSES DISTINTOS")
if "pais" in df.columns:
    print(df["pais"].nunique())

print("\n7. DUPLICIDADES POR PAÍS")
if "pais" in df.columns:
    print(
        df["pais"]
        .value_counts()
        .value_counts()
        .sort_index()
        .to_string()
    )

print("\n8. DUPLICIDADES POR PAÍS E MODELO")
if {"pais", "modelo"}.issubset(df.columns):
    print(
        df.groupby(["pais", "modelo"])
        .size()
        .value_counts()
        .sort_index()
        .to_string()
    )

print("\n9. VALORES AUSENTES")
print(df.isna().sum().to_string())

print("\nINSPEÇÃO CONCLUÍDA.")
