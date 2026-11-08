
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE_DIR / "data" / "processed"
    / "painel_owid_paises_v2.csv"
)

PASTA_SAIDA = BASE_DIR / "data" / "processed"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ARQUIVO)

variaveis = [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita"
]

for coluna in variaveis:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

df = df[df["Year"].between(2000, 2024)].copy()
df = df.sort_values(["Entity", "Year"])

df["log_pib"] = np.where(
    df["pib_per_capita"] > 0,
    np.log(df["pib_per_capita"]),
    np.nan
)

variaveis_diferencas = {
    "delta_mortalidade": "mortalidade_infantil",
    "delta_combustiveis": "combustiveis_limpos",
    "delta_log_pib": "log_pib"
}

for nova_coluna, coluna_original in variaveis_diferencas.items():
    df[nova_coluna] = (
        df.groupby("Entity")[coluna_original].diff()
    )

df["intervalo_anual"] = df.groupby("Entity")["Year"].diff()

df = df[df["intervalo_anual"] == 1].copy()

print("=" * 70)
print("DIAGNÓSTICO DE VALORES EXTREMOS")
print("=" * 70)

colunas = list(variaveis_diferencas.keys())

print("\nESTATÍSTICAS DESCRITIVAS")
print(df[colunas].describe().round(4))

resultados = []

for coluna in colunas:
    serie = df[coluna].dropna()

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    extremos = df[
        (df[coluna] < limite_inferior)
        | (df[coluna] > limite_superior)
    ][["Entity", "Year", coluna]].copy()

    extremos["variavel"] = coluna

    resultados.append(extremos)

    print(f"\n{coluna}")
    print("Limite inferior:", round(limite_inferior, 4))
    print("Limite superior:", round(limite_superior, 4))
    print("Observações extremas:", len(extremos))

if resultados:
    outliers = pd.concat(resultados, ignore_index=True)
else:
    outliers = pd.DataFrame()

arquivo_saida = PASTA_SAIDA / "diagnostico_outliers.csv"
outliers.to_csv(arquivo_saida, index=False, encoding="utf-8-sig")

print("\n10 MAIORES VARIAÇÕES ABSOLUTAS")

for coluna in colunas:
    print(f"\n{coluna}")
    print(
        df.assign(valor_absoluto=df[coluna].abs())
        .nlargest(10, "valor_absoluto")[
            ["Entity", "Year", coluna]
        ].to_string(index=False)
    )

print("\nArquivo gerado:", arquivo_saida)
