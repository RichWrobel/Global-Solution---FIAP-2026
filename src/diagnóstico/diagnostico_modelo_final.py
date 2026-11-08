
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor


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

PASTA_SAIDA = BASE / "data" / "processed" / "diagnostico_final"
PASTA_FIGURAS = BASE / "figuras" / "diagnostico_final"

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. CARREGAMENTO DOS DADOS
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

df = df.dropna(subset=["Entity", "Year"])
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

amostra = amostra.sort_values(["Entity", "Year"]).reset_index(drop=True)

print("\nAMOSTRA DO MODELO")
print(f"Observações: {len(amostra)}")
print(f"Entidades: {amostra['Entity'].nunique()}")
print(f"Período: {amostra['Year'].min()} a {amostra['Year'].max()}")


# ==========================================================
# 4. ESTIMAÇÃO DO MODELO
# ==========================================================

formula = (
    "delta_mortalidade ~ "
    "delta_combustiveis + "
    "delta_log_pib + "
    "C(Year)"
)

modelo = smf.ols(
    formula=formula,
    data=amostra
)

# OLS convencional para diagnósticos.
resultado_ols = modelo.fit()

# Erros-padrão agrupados por entidade para inferência principal.
resultado_cluster = modelo.fit(
    cov_type="cluster",
    cov_kwds={"groups": amostra["Entity"]}
)

amostra["valor_ajustado"] = resultado_ols.fittedvalues
amostra["residuo"] = resultado_ols.resid

amostra["residuo_padronizado"] = (
    resultado_ols.get_influence()
    .resid_studentized_internal
)

print("\nMODELO ESTIMADO")
print(resultado_cluster.summary())


# ==========================================================
# 5. TESTE DE HETEROCEDASTICIDADE
# ==========================================================

bp = het_breuschpagan(
    resultado_ols.resid,
    resultado_ols.model.exog
)

bp_lm, bp_lm_p, bp_f, bp_f_p = bp


# ==========================================================
# 6. NORMALIDADE DOS RESÍDUOS
# ==========================================================

jb_stat, jb_p, assimetria, curtose = jarque_bera(
    resultado_ols.resid
)

# ==========================================================
# 7. AUTOCORRELAÇÃO
# ==========================================================

dw = durbin_watson(resultado_ols.resid)

# Correlação entre resíduos de anos consecutivos
# dentro da mesma entidade.
amostra["residuo_anterior"] = (
    amostra.groupby("Entity")["residuo"].shift(1)
)

amostra["ano_anterior"] = (
    amostra.groupby("Entity")["Year"].shift(1)
)

pares_consecutivos = amostra[
    (amostra["Year"] - amostra["ano_anterior"] == 1)
    & amostra["residuo_anterior"].notna()
].copy()

if len(pares_consecutivos) >= 3:
    correlacao_residuos = pares_consecutivos[
        ["residuo", "residuo_anterior"]
    ].corr().iloc[0, 1]
else:
    correlacao_residuos = np.nan


# ==========================================================
# 8. MULTICOLINEARIDADE
# ==========================================================

# Calcula VIF apenas para as duas variáveis explicativas
# principais, sem incluir as dummies de ano.
X_vif = amostra[
    ["delta_combustiveis", "delta_log_pib"]
].dropna()

vif_resultados = []

for i, coluna in enumerate(X_vif.columns):
    vif_resultados.append({
        "variavel": coluna,
        "VIF": variance_inflation_factor(
            X_vif.values,
            i
        )
    })

vif_df = pd.DataFrame(vif_resultados)


# ==========================================================
# 9. OBSERVAÇÕES COM RESÍDUOS EXTREMOS
# ==========================================================

residuos_extremos = amostra[
    amostra["residuo_padronizado"].abs() > 3
].copy()

colunas_extremos = [
    "Entity",
    "Year",
    "delta_mortalidade",
    "delta_combustiveis",
    "delta_log_pib",
    "residuo",
    "residuo_padronizado",
]

residuos_extremos = residuos_extremos[
    colunas_extremos
].sort_values(
    "residuo_padronizado",
    key=lambda x: x.abs(),
    ascending=False
)


# ==========================================================
# 10. DISTÂNCIA DE COOK
# ==========================================================

influencia = resultado_ols.get_influence()

cook = influencia.cooks_distance[0]

amostra["distancia_cook"] = cook

limite_cook = 4 / len(amostra)

influentes_cook = amostra[
    amostra["distancia_cook"] > limite_cook
][[
    "Entity",
    "Year",
    "distancia_cook",
    "residuo_padronizado",
]].sort_values(
    "distancia_cook",
    ascending=False
)


# ==========================================================
# 11. CONDIÇÃO NUMÉRICA DO MODELO
# ==========================================================

numero_condicao = resultado_ols.condition_number


# ==========================================================
# 12. CONSOLIDAÇÃO DOS DIAGNÓSTICOS
# ==========================================================

metricas = [
    {
        "diagnostico": "Observacoes",
        "estatistica": len(amostra),
        "p_valor": np.nan,
    },
    {
        "diagnostico": "Entidades",
        "estatistica": amostra["Entity"].nunique(),
        "p_valor": np.nan,
    },
    {
        "diagnostico": "Durbin_Watson",
        "estatistica": dw,
        "p_valor": np.nan,
    },
    {
        "diagnostico": "Correlacao_residuos_consecutivos",
        "estatistica": correlacao_residuos,
        "p_valor": np.nan,
    },
    {
        "diagnostico": "Breusch_Pagan_LM",
        "estatistica": bp_lm,
        "p_valor": bp_lm_p,
    },
    {
        "diagnostico": "Breusch_Pagan_F",
        "estatistica": bp_f,
        "p_valor": bp_f_p,
    },
    {
        "diagnostico": "Jarque_Bera",
        "estatistica": jb_stat,
        "p_valor": jb_p,
    },
    {
        "diagnostico": "Assimetria_residuos",
        "estatistica": assimetria,
        "p_valor": np.nan,
    },
    {
        "diagnostico": "Curtose_residuos",
        "estatistica": curtose,
        "p_valor": np.nan,
    },
    {
        "diagnostico": "Numero_condicao",
        "estatistica": numero_condicao,
        "p_valor": np.nan,
    },
    {
        "diagnostico": "Residuos_padronizados_acima_3",
        "estatistica": len(residuos_extremos),
        "p_valor": np.nan,
    },
    {
        "diagnostico": "Observacoes_Cook_acima_4_n",
        "estatistica": len(influentes_cook),
        "p_valor": np.nan,
    },
]

metricas_df = pd.DataFrame(metricas)


# ==========================================================
# 13. GRÁFICOS
# ==========================================================

# Gráfico 1: resíduos versus valores ajustados.
plt.figure(figsize=(10, 6))
plt.scatter(
    amostra["valor_ajustado"],
    amostra["residuo"],
    alpha=0.35
)
plt.axhline(0, linestyle="--")
plt.xlabel("Valores ajustados")
plt.ylabel("Resíduos")
plt.title("Resíduos versus valores ajustados")
plt.tight_layout()
plt.savefig(
    PASTA_FIGURAS / "residuos_valores_ajustados.png",
    dpi=300
)
plt.close()


# Gráfico 2: distribuição dos resíduos.
plt.figure(figsize=(10, 6))
plt.hist(
    amostra["residuo"],
    bins=50,
    edgecolor="black"
)
plt.xlabel("Resíduos")
plt.ylabel("Frequência")
plt.title("Distribuição dos resíduos")
plt.tight_layout()
plt.savefig(
    PASTA_FIGURAS / "distribuicao_residuos.png",
    dpi=300
)
plt.close()


# Gráfico 3: Q-Q plot.
plt.figure(figsize=(8, 8))
stats.probplot(
    amostra["residuo"],
    dist="norm",
    plot=plt
)
plt.title("Q-Q plot dos resíduos")
plt.tight_layout()
plt.savefig(
    PASTA_FIGURAS / "qqplot_residuos.png",
    dpi=300
)
plt.close()


# Gráfico 4: resíduos médios por ano.
residuos_ano = (
    amostra.groupby("Year")["residuo"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))
plt.plot(
    residuos_ano["Year"],
    residuos_ano["residuo"],
    marker="o"
)
plt.axhline(0, linestyle="--")
plt.xlabel("Ano")
plt.ylabel("Resíduo médio")
plt.title("Resíduos médios por ano")
plt.tight_layout()
plt.savefig(
    PASTA_FIGURAS / "residuos_medios_ano.png",
    dpi=300
)
plt.close()


# Gráfico 5: distância de Cook.
plt.figure(figsize=(12, 6))
plt.scatter(
    np.arange(len(amostra)),
    amostra["distancia_cook"],
    alpha=0.5
)
plt.axhline(
    limite_cook,
    linestyle="--",
    label="Referência 4/n"
)
plt.xlabel("Observação")
plt.ylabel("Distância de Cook")
plt.title("Influência das observações — distância de Cook")
plt.legend()
plt.tight_layout()
plt.savefig(
    PASTA_FIGURAS / "distancia_cook.png",
    dpi=300
)
plt.close()


# ==========================================================
# 14. EXPORTAÇÃO
# ==========================================================

metricas_df.to_csv(
    PASTA_SAIDA / "metricas_diagnostico.csv",
    index=False,
    encoding="utf-8-sig"
)

vif_df.to_csv(
    PASTA_SAIDA / "multicolinearidade_vif.csv",
    index=False,
    encoding="utf-8-sig"
)

residuos_extremos.to_csv(
    PASTA_SAIDA / "residuos_extremos.csv",
    index=False,
    encoding="utf-8-sig"
)

influentes_cook.to_csv(
    PASTA_SAIDA / "observacoes_influentes_cook.csv",
    index=False,
    encoding="utf-8-sig"
)

amostra.to_csv(
    PASTA_SAIDA / "residuos_completos.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 15. RESULTADOS NO TERMINAL
# ==========================================================

print("\n" + "=" * 80)
print("DIAGNÓSTICO FINAL DO MODELO")
print("=" * 80)

print("\nMÉTRICAS GERAIS")
print(metricas_df.to_string(index=False))

print("\nMULTICOLINEARIDADE — VIF")
print(vif_df.to_string(index=False))

print("\nRESÍDUOS EXTREMOS")
print(f"Quantidade: {len(residuos_extremos)}")
print(residuos_extremos.head(15).to_string(index=False))

print("\nOBSERVAÇÕES COM DISTÂNCIA DE COOK ACIMA DE 4/n")
print(f"Quantidade: {len(influentes_cook)}")
print(influentes_cook.head(15).to_string(index=False))

print("\n" + "=" * 80)
print("ARQUIVOS GERADOS")
print("=" * 80)

print(f"Tabelas: {PASTA_SAIDA}")
print(f"Gráficos: {PASTA_FIGURAS}")

print("\nDIAGNÓSTICO CONCLUÍDO.")
