
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE / "data" / "processed"
    / "painel_owid_paises_v2.csv"
)

SAIDA = BASE / "data" / "processed"

df = pd.read_csv(ARQUIVO)

variaveis = [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita",
]

for coluna in variaveis:
    df[coluna] = pd.to_numeric(
        df[coluna], errors="coerce"
    )

df["Year"] = pd.to_numeric(
    df["Year"], errors="coerce"
)

df = df[
    df["Year"].between(2000, 2024)
].copy()

df = df.dropna(
    subset=["Entity", "Year"]
)

df = df.sort_values(
    ["Entity", "Year"]
)

# Remover duplicidades, caso existam
if df.duplicated(["Entity", "Year"]).any():
    raise ValueError("Há duplicidades de país/ano.")

# Logaritmo do PIB
df.loc[
    df["pib_per_capita"] <= 0,
    "pib_per_capita"
] = np.nan

df["log_pib_per_capita"] = np.log(
    df["pib_per_capita"]
)

# Calcular primeiras diferenças
colunas = [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "log_pib_per_capita",
]

for coluna in colunas:
    df["delta_" + coluna] = (
        df.groupby("Entity")[coluna].diff()
    )

# Manter apenas anos consecutivos
df["ano_anterior"] = (
    df.groupby("Entity")["Year"].shift(1)
)

df["intervalo_anual"] = (
    df["Year"] - df["ano_anterior"]
)

df = df[df["intervalo_anual"] == 1].copy()

amostra = df.dropna(
    subset=[
        "delta_mortalidade_infantil",
        "delta_combustiveis_limpos",
        "delta_log_pib_per_capita",
    ]
).copy()

print("=" * 70)
print("MODELO DE PRIMEIRAS DIFERENÇAS")
print("=" * 70)

print("Observações:", len(amostra))
print("Entidades:", amostra["Entity"].nunique())
print(
    "Período:",
    int(amostra["Year"].min()),
    "a",
    int(amostra["Year"].max())
)

formula = (
    "delta_mortalidade_infantil ~ "
    "delta_combustiveis_limpos + "
    "delta_log_pib_per_capita + C(Year)"
)

modelo = smf.ols(
    formula=formula,
    data=amostra
).fit(
    cov_type="cluster",
    cov_kwds={"groups": amostra["Entity"]}
)

print("\nRESULTADOS")
print(modelo.summary())

print("\nCOEFICIENTES PRINCIPAIS")

for variavel in [
    "delta_combustiveis_limpos",
    "delta_log_pib_per_capita",
]:
    print(
        f"{variavel}: "
        f"coef={modelo.params[variavel]:.6f} | "
        f"EP={modelo.bse[variavel]:.6f} | "
        f"p={modelo.pvalues[variavel]:.6f}"
    )

resultado = pd.DataFrame({
    "variavel": [
        "delta_combustiveis_limpos",
        "delta_log_pib_per_capita",
    ],
    "coeficiente": [
        modelo.params["delta_combustiveis_limpos"],
        modelo.params["delta_log_pib_per_capita"],
    ],
    "erro_padrao": [
        modelo.bse["delta_combustiveis_limpos"],
        modelo.bse["delta_log_pib_per_capita"],
    ],
    "p_valor": [
        modelo.pvalues["delta_combustiveis_limpos"],
        modelo.pvalues["delta_log_pib_per_capita"],
    ],
})

resultado.to_csv(
    SAIDA / "resultado_primeiras_diferencas.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nArquivo gerado:",
    SAIDA / "resultado_primeiras_diferencas.csv"
)
