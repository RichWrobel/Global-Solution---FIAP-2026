
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


# ==========================================================
# 1. CAMINHOS
# ==========================================================

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE
    / "data"
    / "processed"
    / "painel_owid_paises_v2.csv"
)

PASTA_SAIDA = BASE / "data" / "processed"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 2. CARREGAMENTO E PREPARAÇÃO
# ==========================================================

df = pd.read_csv(ARQUIVO)

colunas = [
    "Entity",
    "Year",
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita",
]

faltantes = [c for c in colunas if c not in df.columns]

if faltantes:
    raise ValueError(
        f"Colunas não encontradas no painel: {faltantes}"
    )

df = df[colunas].copy()

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

for coluna in [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita",
]:
    df[coluna] = pd.to_numeric(
        df[coluna],
        errors="coerce"
    )

df = df.dropna(subset=["Entity", "Year"])
df = df.sort_values(["Entity", "Year"])

# Evita logaritmo de valores nulos ou negativos.
df.loc[df["pib_per_capita"] <= 0, "pib_per_capita"] = pd.NA

df["log_pib"] = __import__("numpy").log(df["pib_per_capita"])

# ==========================================================
# 3. PRIMEIRAS DIFERENÇAS
# ==========================================================

grupo = df.groupby("Entity", sort=False)

df["intervalo_anual"] = grupo["Year"].diff()

df["delta_mortalidade"] = grupo[
    "mortalidade_infantil"
].diff()

df["delta_combustiveis"] = grupo[
    "combustiveis_limpos"
].diff()

df["delta_log_pib"] = grupo["log_pib"].diff()

# Considera somente variações entre anos consecutivos.
amostra = df[
    df["intervalo_anual"].eq(1)
].copy()

amostra = amostra.dropna(
    subset=[
        "delta_mortalidade",
        "delta_combustiveis",
        "delta_log_pib",
        "Entity",
        "Year",
    ]
)

print("\nAMOSTRA UTILIZADA")
print(f"Observações: {len(amostra)}")
print(f"Entidades: {amostra['Entity'].nunique()}")
print(
    f"Período: {int(amostra['Year'].min())} "
    f"a {int(amostra['Year'].max())}"
)

# ==========================================================
# 4. ESTIMAÇÃO DOS MODELOS
# ==========================================================

resultados = []
limites = []

def estimar_modelo(dados, nome):
    modelo = smf.ols(
        formula=(
            "delta_mortalidade ~ "
            "delta_combustiveis + "
            "delta_log_pib + "
            "C(Year)"
        ),
        data=dados
    )

    resultado = modelo.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": dados["Entity"]
        }
    )

    resultados.append({
        "amostra": nome,
        "observacoes": int(resultado.nobs),
        "entidades": dados["Entity"].nunique(),
        "coef_combustiveis": resultado.params[
            "delta_combustiveis"
        ],
        "p_combustiveis": resultado.pvalues[
            "delta_combustiveis"
        ],
        "coef_log_pib": resultado.params[
            "delta_log_pib"
        ],
        "p_log_pib": resultado.pvalues[
            "delta_log_pib"
        ],
        "r2": resultado.rsquared,
    })

    return resultado


# Modelo original.
modelo_original = estimar_modelo(
    amostra.copy(),
    "Original"
)

# ==========================================================
# 5. WINSORIZAÇÃO DA VARIAÇÃO DA MORTALIDADE
# ==========================================================

for percentual, nome in [
    (0.01, "Winsorizado_1_99"),
    (0.025, "Winsorizado_2_5_97_5"),
]:

    dados = amostra.copy()

    limite_inferior = dados[
        "delta_mortalidade"
    ].quantile(percentual)

    limite_superior = dados[
        "delta_mortalidade"
    ].quantile(1 - percentual)

    dados["delta_mortalidade"] = dados[
        "delta_mortalidade"
    ].clip(
        lower=limite_inferior,
        upper=limite_superior
    )

    limites.append({
        "modelo": nome,
        "percentil_inferior": percentual * 100,
        "percentil_superior": (1 - percentual) * 100,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
    })

    estimar_modelo(dados, nome)


# ==========================================================
# 6. RESULTADOS
# ==========================================================

resultado_final = pd.DataFrame(resultados)

print("\n" + "=" * 90)
print("COMPARAÇÃO DOS MODELOS — WINSORIZAÇÃO")
print("=" * 90)

print(
    resultado_final.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)

# ==========================================================
# 7. EXPORTAÇÃO
# ==========================================================

arquivo_resultados = (
    PASTA_SAIDA / "comparacao_winsorizacao.csv"
)

arquivo_limites = (
    PASTA_SAIDA / "limites_winsorizacao.csv"
)

resultado_final.to_csv(
    arquivo_resultados,
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(limites).to_csv(
    arquivo_limites,
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos gerados:")
print(arquivo_resultados)
print(arquivo_limites)
