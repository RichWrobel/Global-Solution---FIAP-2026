
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE
    / "data"
    / "processed"
    / "painel_owid_paises_v2.csv"
)

df = pd.read_csv(ARQUIVO)

print("=" * 75)
print("INSPEÇÃO DA BASE PARA O DASHBOARD")
print("=" * 75)

print("\n1. DIMENSÕES DA BASE")
print(f"Registros: {len(df):,}")
print(f"Colunas: {len(df.columns)}")
print(f"Países/entidades: {df['Entity'].nunique()}")
print(f"Período: {df['Year'].min()}–{df['Year'].max()}")

print("\n2. COLUNAS DISPONÍVEIS")
for coluna in df.columns:
    print(f"- {coluna}")

print("\n3. TIPOS DE DADOS")
print(df.dtypes.to_string())

print("\n4. INDICADORES E COBERTURA")
resumo = pd.DataFrame({
    "valores_validos": df.count(),
    "ausentes": df.isna().sum(),
    "percentual_ausente": (df.isna().mean() * 100).round(2)
})

print(resumo.to_string())

print("\n5. AMOSTRA DE REGISTROS")
print(df.head(10).to_string(index=False))

print("\n6. ÚLTIMO ANO DISPONÍVEL POR INDICADOR")

for coluna in df.columns:
    if coluna not in ["Entity", "Code", "Year"]:
        validos = df.loc[df[coluna].notna(), "Year"]

        if not validos.empty:
            print(f"{coluna}: {int(validos.max())}")

print("\nINSPEÇÃO CONCLUÍDA.")
