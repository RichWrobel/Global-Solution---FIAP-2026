
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

colunas = [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita",
    "escolaridade",
]

for coluna in colunas:
    df[coluna] = pd.to_numeric(
        df[coluna], errors="coerce"
    )

df["Year"] = pd.to_numeric(
    df["Year"], errors="coerce"
)

df = df[df["Year"].between(2000, 2024)].copy()

df.loc[
    df["pib_per_capita"] <= 0,
    "pib_per_capita"
] = np.nan

df["log_pib_per_capita"] = np.log(
    df["pib_per_capita"]
)

formula_base = (
    "mortalidade_infantil ~ combustiveis_limpos "
    "+ log_pib_per_capita + C(Entity) + C(Year)"
)

formula_escolaridade = (
    "mortalidade_infantil ~ combustiveis_limpos "
    "+ log_pib_per_capita + escolaridade "
    "+ C(Entity) + C(Year)"
)

def estimar(base, formula, nome):
    print("\n" + "=" * 70)
    print(nome)
    print("=" * 70)

    modelo = smf.ols(
        formula=formula,
        data=base
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": base["Entity"]}
    )

    print("Observações:", int(modelo.nobs))
    print("Entidades:", base["Entity"].nunique())
    print("Anos:", base["Year"].nunique())

    print("\nCoeficientes:")
    for variavel in [
        "combustiveis_limpos",
        "log_pib_per_capita",
        "escolaridade",
    ]:
        if variavel in modelo.params.index:
            print(
                f"{variavel}: "
                f"coef={modelo.params[variavel]:.6f} | "
                f"EP={modelo.bse[variavel]:.6f} | "
                f"p={modelo.pvalues[variavel]:.6f}"
            )

    print("\nR²:", round(modelo.rsquared, 4))

    return modelo

# Amostra principal: 3 variáveis completas
amostra_principal = df.dropna(
    subset=[
        "mortalidade_infantil",
        "combustiveis_limpos",
        "log_pib_per_capita",
    ]
).copy()

# Amostra comum: exige também escolaridade
amostra_comum = amostra_principal.dropna(
    subset=["escolaridade"]
).copy()

# Modelo 1: amostra principal
m1 = estimar(
    amostra_principal,
    formula_base,
    "MODELO 1 — AMOSTRA PRINCIPAL"
)

# Modelo 2: mesma especificação, amostra comum
m2 = estimar(
    amostra_comum,
    formula_base,
    "MODELO 2 — AMOSTRA COMUM, SEM ESCOLARIDADE"
)

# Modelo 3: mesma amostra, incluindo escolaridade
m3 = estimar(
    amostra_comum,
    formula_escolaridade,
    "MODELO 3 — AMOSTRA COMUM, COM ESCOLARIDADE"
)

# Comparação
comparacao = pd.DataFrame([
    {
        "modelo": "Principal",
        "observacoes": int(m1.nobs),
        "entidades": amostra_principal["Entity"].nunique(),
        "anos": amostra_principal["Year"].nunique(),
        "coef_combustiveis": m1.params["combustiveis_limpos"],
        "p_combustiveis": m1.pvalues["combustiveis_limpos"],
        "r2": m1.rsquared,
    },
    {
        "modelo": "Amostra comum sem escolaridade",
        "observacoes": int(m2.nobs),
        "entidades": amostra_comum["Entity"].nunique(),
        "anos": amostra_comum["Year"].nunique(),
        "coef_combustiveis": m2.params["combustiveis_limpos"],
        "p_combustiveis": m2.pvalues["combustiveis_limpos"],
        "r2": m2.rsquared,
    },
    {
        "modelo": "Amostra comum com escolaridade",
        "observacoes": int(m3.nobs),
        "entidades": amostra_comum["Entity"].nunique(),
        "anos": amostra_comum["Year"].nunique(),
        "coef_combustiveis": m3.params["combustiveis_limpos"],
        "p_combustiveis": m3.pvalues["combustiveis_limpos"],
        "r2": m3.rsquared,
    },
])

comparacao.to_csv(
    SAIDA / "comparacao_modelos_v3.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("COMPARAÇÃO FINAL")
print("=" * 70)
print(comparacao.to_string(index=False))

print("\nArquivo salvo:", SAIDA / "comparacao_modelos_v3.csv")
