
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PASTA_RAW = BASE / "data" / "raw"
ARQUIVO_PAINEL = (
    BASE / "data" / "processed" / "painel_owid_integrado.csv"
)

print("=" * 65)
print("DIAGNÓSTICO DE ENTIDADES E CÓDIGOS")
print("=" * 65)

# 1. Verificar estrutura dos arquivos originais
for caminho in sorted(PASTA_RAW.glob("*.csv")):
    df = pd.read_csv(caminho, nrows=5)

    print(f"\nARQUIVO: {caminho.name}")
    print("Colunas:", list(df.columns))

    colunas = [
        c for c in ["Entity", "Code", "Year"]
        if c in df.columns
    ]

    if colunas:
        print(df[colunas].head(5).to_string(index=False))

# 2. Listar entidades do painel integrado
painel = pd.read_csv(ARQUIVO_PAINEL)

print("\n" + "=" * 65)
print("ENTIDADES PRESENTES NO PAINEL")
print("=" * 65)

entidades = sorted(painel["Entity"].dropna().unique())

print("Total:", len(entidades))

for entidade in entidades:
    print(entidade)

print("\nDiagnóstico concluído.")
