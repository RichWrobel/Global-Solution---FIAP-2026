
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from scipy.stats import pearsonr


# ==========================================================
# 1. CAMINHOS
# ==========================================================

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE / "data" / "processed"
    / "painel_owid_paises_v2.csv"
)

PASTA = (
    BASE / "data" / "processed"
    / "diagnostico_fatores_comuns"
)

PASTA.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CARREGAMENTO
# ==========================================================

print("\nCARREGANDO DADOS...")

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
    raise ValueError(f"Colunas ausentes: {faltantes}")

df = df[colunas].copy()

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

for coluna in [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita",
]:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

df = df.dropna(subset=["Entity", "Year"]).copy()
df = df.sort_values(["Entity", "Year"])

df.loc[df["pib_per_capita"] <= 0, "pib_per_capita"] = np.nan

df["log_pib"] = np.log(df["pib_per_capita"])


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

amostra = df[
    df["intervalo_anual"].eq(1)
].copy()

amostra = amostra.dropna(
    subset=[
        "delta_mortalidade",
        "delta_combustiveis",
        "delta_log_pib",
    ]
).copy()

amostra["Year"] = amostra["Year"].astype(int)

amostra = amostra.sort_values(
    ["Entity", "Year"]
).reset_index(drop=True)

print("\nAMOSTRA PRINCIPAL")
print(f"Observações: {len(amostra)}")
print(f"Países: {amostra['Entity'].nunique()}")
print(
    f"Período: {amostra['Year'].min()} "
    f"a {amostra['Year'].max()}"
)


# ==========================================================
# 4. MÉDIAS TRANSVERSAIS ANUAIS
# ==========================================================

variaveis = [
    "delta_mortalidade",
    "delta_combustiveis",
    "delta_log_pib",
]

medias_anuais = (
    amostra.groupby("Year")[variaveis]
    .agg(["mean", "std", "count"])
)

medias_anuais.columns = [
    f"{variavel}_{estatistica}"
    for variavel, estatistica in medias_anuais.columns
]

medias_anuais = medias_anuais.reset_index()

medias_anuais.to_csv(
    PASTA / "medias_transversais_anuais.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 5. MÉDIAS LEAVE-ONE-OUT
# ==========================================================

# Para cada país e ano, calcula a média dos demais países.
# O próprio país não participa da média utilizada em sua linha.

for variavel, nome in [
    ("delta_mortalidade", "media_loo_mortalidade"),
    ("delta_combustiveis", "media_loo_combustiveis"),
    ("delta_log_pib", "media_loo_log_pib"),
]:

    soma = amostra.groupby("Year")[variavel].transform("sum")

    contagem = amostra.groupby("Year")[variavel].transform("count")

    amostra[nome] = (
        (soma - amostra[variavel])
        / (contagem - 1)
    )

if amostra[
    [
        "media_loo_mortalidade",
        "media_loo_combustiveis",
        "media_loo_log_pib",
    ]
].isna().any().any():

    raise ValueError(
        "Existem médias leave-one-out ausentes."
    )

print("\nMÉDIAS LEAVE-ONE-OUT")
print("Calculadas excluindo o próprio país de cada observação.")


# ==========================================================
# 6. ESTIMAÇÃO DOS MODELOS
# ==========================================================

# Modelo 1: especificação de referência, com efeitos de ano.

formula_1 = (
    "delta_mortalidade ~ "
    "delta_combustiveis + "
    "delta_log_pib + "
    "C(Year)"
)

modelo_1 = smf.ols(
    formula_1,
    data=amostra
).fit(
    cov_type="cluster",
    cov_kwds={"groups": amostra["Entity"]}
)


# Modelo 2: sem efeitos fixos de ano.
# Serve para verificar a importância dos controles temporais.

formula_2 = (
    "delta_mortalidade ~ "
    "delta_combustiveis + "
    "delta_log_pib"
)

modelo_2 = smf.ols(
    formula_2,
    data=amostra
).fit(
    cov_type="cluster",
    cov_kwds={"groups": amostra["Entity"]}
)


# Modelo 3: especificação exploratória com médias transversais.
# Não incluir C(Year), pois as médias leave-one-out variam
# somente entre anos e seriam colineares com as dummies anuais.

formula_3 = (
    "delta_mortalidade ~ "
    "delta_combustiveis + "
    "delta_log_pib + "
    "media_loo_mortalidade + "
    "media_loo_combustiveis + "
    "media_loo_log_pib"
)

modelo_3 = smf.ols(
    formula_3,
    data=amostra
).fit(
    cov_type="cluster",
    cov_kwds={"groups": amostra["Entity"]}
)


# ==========================================================
# 7. TESTE CONJUNTO DAS MÉDIAS TRANSVERSAIS
# ==========================================================

teste_conjunto = modelo_3.wald_test(
    "media_loo_mortalidade = 0, "
    "media_loo_combustiveis = 0, "
    "media_loo_log_pib = 0",
    scalar=True
)


# ==========================================================
# 8. COMPARAÇÃO DOS COEFICIENTES
# ==========================================================

modelos = [
    ("Modelo 1 - efeitos de ano", modelo_1),
    ("Modelo 2 - sem efeitos de ano", modelo_2),
    ("Modelo 3 - médias transversais", modelo_3),
]

linhas = []

for nome, resultado in modelos:

    linhas.append({
        "modelo": nome,
        "observacoes": int(resultado.nobs),
        "R2": resultado.rsquared,
        "coef_combustiveis": resultado.params.get(
            "delta_combustiveis", np.nan
        ),
        "erro_combustiveis": resultado.bse.get(
            "delta_combustiveis", np.nan
        ),
        "p_combustiveis": resultado.pvalues.get(
            "delta_combustiveis", np.nan
        ),
        "coef_log_pib": resultado.params.get(
            "delta_log_pib", np.nan
        ),
        "erro_log_pib": resultado.bse.get(
            "delta_log_pib", np.nan
        ),
        "p_log_pib": resultado.pvalues.get(
            "delta_log_pib", np.nan
        ),
    })

comparacao = pd.DataFrame(linhas)

comparacao.to_csv(
    PASTA / "comparacao_modelos_fatores_comuns.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 9. EXPORTAÇÃO DOS COEFICIENTES DO MODELO 3
# ==========================================================

coeficientes = pd.DataFrame({
    "variavel": modelo_3.params.index,
    "coeficiente": modelo_3.params.values,
    "erro_padrao_cluster": modelo_3.bse.values,
    "p_valor": modelo_3.pvalues.values,
})

coeficientes.to_csv(
    PASTA / "coeficientes_modelo_fatores_comuns.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 10. TESTE CONJUNTO
# ==========================================================

teste_df = pd.DataFrame([{
    "estatistica_wald": float(teste_conjunto.statistic),
    "p_valor": float(teste_conjunto.pvalue),
    "hipotese_nula": (
        "As três médias transversais são conjuntamente iguais a zero"
    ),
}])

teste_df.to_csv(
    PASTA / "teste_conjunto_medias_transversais.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 11. CORRELAÇÕES DAS MÉDIAS ANUAIS
# ==========================================================

correlacoes = (
    medias_anuais[
        [
            "delta_mortalidade_mean",
            "delta_combustiveis_mean",
            "delta_log_pib_mean",
        ]
    ]
    .corr(method="pearson")
)

correlacoes.to_csv(
    PASTA / "correlacoes_medias_anuais.csv",
    encoding="utf-8-sig"
)


# ==========================================================
# 12. RESUMO E INTEGRIDADE
# ==========================================================

integridade = pd.DataFrame([{
    "observacoes": len(amostra),
    "paises": amostra["Entity"].nunique(),
    "ano_inicial": amostra["Year"].min(),
    "ano_final": amostra["Year"].max(),
    "medias_loo_sem_ausencias": (
        not amostra[
            [
                "media_loo_mortalidade",
                "media_loo_combustiveis",
                "media_loo_log_pib",
            ]
        ].isna().any().any()
    ),
}])

integridade.to_csv(
    PASTA / "integridade_amostra.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 13. RESULTADOS NO TERMINAL
# ==========================================================

print("\n" + "=" * 90)
print("DIAGNÓSTICO DE FATORES COMUNS")
print("=" * 90)

print("\nCOMPARAÇÃO DOS MODELOS")

print(
    comparacao.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nTESTE CONJUNTO DAS MÉDIAS TRANSVERSAIS")

print(teste_df.to_string(index=False))

print("\nCORRELAÇÕES ENTRE MÉDIAS ANUAIS")

print(correlacoes.to_string())

print("\nINTEGRIDADE DA AMOSTRA")

print(integridade.to_string(index=False))

print("\nARQUIVOS GERADOS:")
print(PASTA)

print("\nDIAGNÓSTICO CONCLUÍDO.")
