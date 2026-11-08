
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE / "data" / "processed" / "modelo_ccemg"
    / "coeficientes_por_pais.csv"
)

SAIDA = (
    BASE / "data" / "processed"
    / "sensibilidade_ccemg_corrigida"
)

SAIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ARQUIVO)

# Manter somente o modelo CCEMG
ccemg = df[df["modelo"] == "CCEMG"].copy()

# Garantir tipos numéricos
colunas = [
    "observacoes",
    "coef_combustiveis",
    "coef_log_pib",
    "limite_minimo_observacoes"
]

for coluna in colunas:
    ccemg[coluna] = pd.to_numeric(
        ccemg[coluna],
        errors="coerce"
    )

ccemg = ccemg.dropna(subset=colunas)

print("=" * 80)
print("SENSIBILIDADE CORRIGIDA - SOMENTE CCEMG")
print("=" * 80)

resultados = []

for limite in sorted(
    ccemg["limite_minimo_observacoes"].unique()
):

    amostra = ccemg[
        ccemg["limite_minimo_observacoes"] == limite
    ].copy()

    # Verificar se há mais de uma linha por país
    duplicados = amostra["pais"].duplicated().sum()

    if duplicados > 0:
        print(
            f"\nATENÇÃO: {duplicados} duplicidades "
            f"no limite {limite}."
        )

        amostra = amostra.drop_duplicates(
            subset="pais",
            keep="first"
        )

    for variavel in [
        "coef_combustiveis",
        "coef_log_pib"
    ]:

        valores = amostra[variavel].dropna()

        resultados.append({
            "modelo": "CCEMG",
            "limite_minimo": int(limite),
            "variavel": variavel,
            "paises": valores.size,
            "media": valores.mean(),
            "mediana": valores.median(),
            "desvio_padrao": valores.std(),
            "minimo": valores.min(),
            "maximo": valores.max()
        })

resultado = pd.DataFrame(resultados)

resultado.to_csv(
    SAIDA / "sensibilidade_ccemg_por_limite.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n1. QUANTIDADE DE PAÍSES POR LIMITE")

print(
    ccemg.groupby("limite_minimo_observacoes")["pais"]
    .nunique()
    .to_string()
)

print("\n2. RESULTADOS")

print(
    resultado.round(6).to_string(index=False)
)

print("\n3. VERIFICAÇÃO DE DUPLICIDADES")

print(
    ccemg.groupby("limite_minimo_observacoes")
    .agg(
        linhas=("pais", "size"),
        paises=("pais", "nunique")
    )
    .to_string()
)

print("\nARQUIVO GERADO:")
print(SAIDA / "sensibilidade_ccemg_por_limite.csv")

print("\nANÁLISE CONCLUÍDA.")
