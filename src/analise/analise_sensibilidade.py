
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

BASE_DIR = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE_DIR / "data" / "processed"
    / "painel_owid_paises_v2.csv"
)

SAIDA = (
    BASE_DIR / "data" / "processed"
    / "comparacao_sensibilidade.csv"
)

df = pd.read_csv(ARQUIVO)

for col in [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["Year"].between(2000, 2024)].copy()
df = df.sort_values(["Entity", "Year"])

df["log_pib"] = np.where(
    df["pib_per_capita"] > 0,
    np.log(df["pib_per_capita"]),
    np.nan
)

df["delta_mortalidade"] = (
    df.groupby("Entity")["mortalidade_infantil"].diff()
)

df["delta_combustiveis"] = (
    df.groupby("Entity")["combustiveis_limpos"].diff()
)

df["delta_log_pib"] = (
    df.groupby("Entity")["log_pib"].diff()
)

df["intervalo_anual"] = df.groupby("Entity")["Year"].diff()

# Amostra de primeiras diferenças
dados = df[df["intervalo_anual"] == 1].copy()

dados = dados.dropna(subset=[
    "delta_mortalidade",
    "delta_combustiveis",
    "delta_log_pib"
])

# Limites IQR
limites = {}

for coluna in [
    "delta_mortalidade",
    "delta_combustiveis",
    "delta_log_pib"
]:
    q1 = dados[coluna].quantile(0.25)
    q3 = dados[coluna].quantile(0.75)
    iqr = q3 - q1

    limites[coluna] = (
        q1 - 1.5 * iqr,
        q3 + 1.5 * iqr
    )

def dentro_limite(df, coluna):
    inferior, superior = limites[coluna]
    return df[coluna].between(inferior, superior)

# Três amostras comparáveis
amostras = {
    "Original": dados.copy(),

    "Sem extremos de mortalidade": dados[
        dentro_limite(dados, "delta_mortalidade")
    ].copy(),

    "Sem extremos nas tres variaveis": dados[
        dentro_limite(dados, "delta_mortalidade")
        & dentro_limite(dados, "delta_combustiveis")
        & dentro_limite(dados, "delta_log_pib")
    ].copy()
}

resultados = []

for nome, amostra in amostras.items():

    modelo = smf.ols(
        "delta_mortalidade ~ delta_combustiveis "
        "+ delta_log_pib + C(Year)",
        data=amostra
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": amostra["Entity"]}
    )

    resultados.append({
        "amostra": nome,
        "observacoes": int(modelo.nobs),
        "entidades": amostra["Entity"].nunique(),
        "coef_combustiveis": modelo.params[
            "delta_combustiveis"
        ],
        "p_combustiveis": modelo.pvalues[
            "delta_combustiveis"
        ],
        "coef_log_pib": modelo.params[
            "delta_log_pib"
        ],
        "p_log_pib": modelo.pvalues[
            "delta_log_pib"
        ],
        "r2": modelo.rsquared
    })

resultado = pd.DataFrame(resultados)

resultado.to_csv(
    SAIDA,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 70)
print("ANÁLISE DE SENSIBILIDADE")
print("=" * 70)

print(resultado.to_string(index=False))

print("\nArquivo gerado:", SAIDA)
