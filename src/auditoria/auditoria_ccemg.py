
from pathlib import Path
import pandas as pd
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]

PASTA = RAIZ / "data" / "processed" / "modelo_ccemg"

SAIDA = (
    RAIZ / "data" / "processed" /
    "auditoria_ccemg"
)

SAIDA.mkdir(parents=True, exist_ok=True)

MINIMO = 20
MODELO = "CCEMG"

arquivo_diag = (
    PASTA / f"diagnostico_{MODELO}_minimo_{MINIMO}.csv"
)

arquivo_coef = PASTA / "coeficientes_por_pais.csv"

diag = pd.read_csv(arquivo_diag)
coef = pd.read_csv(arquivo_coef)

coef = coef[
    (coef["modelo"] == MODELO)
    & (coef["limite_minimo_observacoes"] == MINIMO)
].copy()

print("=" * 80)
print("AUDITORIA DO CCEMG")
print("=" * 80)

print("\n1. SITUAÇÃO DAS ESTIMAÇÕES")

if "situacao" in diag.columns:
    print(diag["situacao"].value_counts(dropna=False).to_string())
else:
    print("Coluna situacao não encontrada.")

print("\n2. DIAGNÓSTICO NUMÉRICO")

if not diag.empty and "numero_condicao" in diag.columns:
    print(
        diag["numero_condicao"]
        .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .to_string()
    )

    print("\nPaíses com maior número de condição:")
    print(
        diag.sort_values(
            "numero_condicao", ascending=False
        ).head(15).to_string(index=False)
    )

print("\n3. DISTRIBUIÇÃO DOS COEFICIENTES INDIVIDUAIS")

variaveis = [
    "coef_combustiveis",
    "coef_log_pib",
]

for coluna in variaveis:
    print(f"\n{coluna}")
    print(
        coef[coluna]
        .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
        .to_string()
    )

print("\n4. PAÍSES COM COEFICIENTES EXTREMOS")

for coluna in variaveis:
    print(f"\nMaiores valores de {coluna}:")
    print(
        coef.nlargest(10, coluna)[
            ["pais", "observacoes", coluna]
        ].to_string(index=False)
    )

    print(f"\nMenores valores de {coluna}:")
    print(
        coef.nsmallest(10, coluna)[
            ["pais", "observacoes", coluna]
        ].to_string(index=False)
    )

print("\n5. CONTAGEM DE PAÍSES")

print(f"Países diagnosticados: {diag['pais'].nunique()}")
print(f"Países estimados: {coef['pais'].nunique()}")

diagnostico_final = diag.merge(
    coef[["pais", "coef_combustiveis", "coef_log_pib"]],
    on="pais",
    how="left",
)

diagnostico_final.to_csv(
    SAIDA / "auditoria_paises_ccemg.csv",
    index=False,
)

print("\nARQUIVO GERADO:")
print(SAIDA / "auditoria_paises_ccemg.csv")

print("\nAUDITORIA CONCLUÍDA.")
