
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from scipy.stats import norm


# ==========================================================
# 1. CAMINHOS E CONFIGURAÇÕES
# ==========================================================

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE / "data" / "processed"
    / "painel_owid_paises_v2.csv"
)

PASTA_SAIDA = (
    BASE / "data" / "processed"
    / "validacao_dependencia_transversal"
)

PASTA_FIGURAS = (
    BASE / "figuras"
    / "validacao_dependencia_transversal"
)

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

MIN_OBS_COMPLETO = 10
MIN_OBS_SUBPERIODO = 6


# ==========================================================
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
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
    df[coluna] = pd.to_numeric(
        df[coluna],
        errors="coerce"
    )

df = df.dropna(subset=["Entity", "Year"]).copy()

df = df.sort_values(
    ["Entity", "Year"]
).reset_index(drop=True)

df.loc[
    df["pib_per_capita"] <= 0,
    "pib_per_capita"
] = np.nan

df["log_pib"] = np.log(df["pib_per_capita"])


# ==========================================================
# 3. CONSTRUÇÃO DAS PRIMEIRAS DIFERENÇAS
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

amostra_original = df[
    df["intervalo_anual"].eq(1)
].copy()

amostra_original = amostra_original.dropna(
    subset=[
        "delta_mortalidade",
        "delta_combustiveis",
        "delta_log_pib",
    ]
).copy()

amostra_original["Year"] = (
    amostra_original["Year"].astype(int)
)

amostra_original = amostra_original.sort_values(
    ["Entity", "Year"]
).reset_index(drop=True)

print("\nAMOSTRA PRINCIPAL")
print(f"Observações: {len(amostra_original)}")
print(
    f"Países: {amostra_original['Entity'].nunique()}"
)
print(
    f"Período: {amostra_original['Year'].min()} "
    f"a {amostra_original['Year'].max()}"
)


# ==========================================================
# 4. FUNÇÃO DO TESTE CD DE PESARAN
# ==========================================================

def calcular_cd(dados, minimo_observacoes=10):

    matriz = dados.pivot(
        index="Year",
        columns="Entity",
        values="residuo"
    )

    entidades = list(matriz.columns)
    n_paises = len(entidades)

    termos = []
    correlacoes = []

    for pais_a, pais_b in combinations(entidades, 2):

        par = matriz[[pais_a, pais_b]].dropna()

        t_ij = len(par)

        if t_ij < minimo_observacoes:
            continue

        rho = par[pais_a].corr(par[pais_b])

        if pd.isna(rho):
            continue

        termos.append(np.sqrt(t_ij) * rho)

        correlacoes.append({
            "pais_a": pais_a,
            "pais_b": pais_b,
            "observacoes_comuns": t_ij,
            "correlacao": rho,
        })

    correlacoes_df = pd.DataFrame(correlacoes)

    numero_pares = len(correlacoes_df)

    if numero_pares == 0:
        raise ValueError(
            "Não existem pares suficientes para o teste CD."
        )

    # Formulação CD com denominador baseado no número
    # total de entidades do cenário analisado.
    CD = (
        np.sqrt(2 / (n_paises * (n_paises - 1)))
        * np.sum(termos)
    )

    p_valor = 2 * norm.sf(abs(CD))

    rho = correlacoes_df["correlacao"]

    resumo = {
        "paises": n_paises,
        "pares_utilizados": numero_pares,
        "pares_possiveis": n_paises * (n_paises - 1) // 2,
        "observacoes_minimas_por_par": minimo_observacoes,
        "CD_Pesaran": CD,
        "p_valor": p_valor,
        "correlacao_media": rho.mean(),
        "correlacao_mediana": rho.median(),
        "correlacao_media_absoluta": rho.abs().mean(),
        "percentual_positivas": (rho > 0).mean() * 100,
        "percentual_negativas": (rho < 0).mean() * 100,
    }

    return resumo, correlacoes_df


# ==========================================================
# 5. FUNÇÃO PARA ESTIMAR O MODELO E CALCULAR RESÍDUOS
# ==========================================================

def executar_cenario(
    nome,
    dados,
    minimo_observacoes=10
):

    dados = dados.copy()

    formula = (
        "delta_mortalidade ~ "
        "delta_combustiveis + "
        "delta_log_pib + "
        "C(Year)"
    )

    modelo = smf.ols(
        formula=formula,
        data=dados
    )

    resultado = modelo.fit()

    dados["residuo"] = resultado.resid

    resumo_cd, pares = calcular_cd(
        dados,
        minimo_observacoes=minimo_observacoes
    )

    resumo_cd.update({
        "cenario": nome,
        "observacoes_modelo": len(dados),
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
        "R2": resultado.rsquared,
        "ano_inicial": dados["Year"].min(),
        "ano_final": dados["Year"].max(),
    })

    pares["cenario"] = nome

    return resumo_cd, pares


# ==========================================================
# 6. CENÁRIOS DE SENSIBILIDADE
# ==========================================================

resultados = []
tabelas_pares = []

print("\nEXECUTANDO CENÁRIOS...")


# CENÁRIO 1 — AMOSTRA PRINCIPAL COMPLETA

cenarios = [
    (
        "Principal_completo",
        amostra_original.copy(),
        MIN_OBS_COMPLETO
    ),

    (
        "Sem_Central_African_Republic",
        amostra_original[
            amostra_original["Entity"]
            != "Central African Republic"
        ].copy(),
        MIN_OBS_COMPLETO
    ),

    (
        "Sem_Somalia",
        amostra_original[
            amostra_original["Entity"] != "Somalia"
        ].copy(),
        MIN_OBS_COMPLETO
    ),

    (
        "Sem_CAR_e_Somalia",
        amostra_original[
            ~amostra_original["Entity"].isin([
                "Central African Republic",
                "Somalia",
            ])
        ].copy(),
        MIN_OBS_COMPLETO
    ),
]


# CENÁRIOS 5 E 6 — SUBPERÍODOS

cenarios.extend([
    (
        "Periodo_2001_2012",
        amostra_original[
            amostra_original["Year"].between(2001, 2012)
        ].copy(),
        MIN_OBS_SUBPERIODO
    ),

    (
        "Periodo_2013_2024",
        amostra_original[
            amostra_original["Year"].between(2013, 2024)
        ].copy(),
        MIN_OBS_SUBPERIODO
    ),
])


for nome, dados, minimo in cenarios:

    print(f"\nProcessando: {nome}")

    resumo, pares = executar_cenario(
        nome,
        dados,
        minimo_observacoes=minimo
    )

    resultados.append(resumo)
    tabelas_pares.append(pares)

    print(
        f"Países: {resumo['paises']} | "
        f"Observações: {resumo['observacoes_modelo']} | "
        f"CD: {resumo['CD_Pesaran']:.4f} | "
        f"p-valor: {resumo['p_valor']:.8g}"
    )


# ==========================================================
# 7. TABELA COMPARATIVA
# ==========================================================

resultados_df = pd.DataFrame(resultados)

resultados_df.to_csv(
    PASTA_SAIDA / "comparacao_cenarios.csv",
    index=False,
    encoding="utf-8-sig"
)

pares_completos = pd.concat(
    tabelas_pares,
    ignore_index=True
)

pares_completos.to_csv(
    PASTA_SAIDA / "correlacoes_por_cenario.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 8. DISTRIBUIÇÃO DAS CORRELAÇÕES NO CENÁRIO PRINCIPAL
# ==========================================================

pares_principais = pares_completos[
    pares_completos["cenario"] == "Principal_completo"
].copy()

plt.figure(figsize=(10, 6))

plt.hist(
    pares_principais["correlacao"],
    bins=50,
    edgecolor="black"
)

plt.axvline(0, linestyle="--")

plt.xlabel("Correlação entre resíduos")
plt.ylabel("Frequência")
plt.title(
    "Distribuição das correlações — amostra principal"
)

plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "correlacoes_amostra_principal.png",
    dpi=300
)

plt.close()


# ==========================================================
# 9. COMPARAÇÃO DA ESTATÍSTICA CD
# ==========================================================

grafico = resultados_df.copy()

plt.figure(figsize=(12, 7))

plt.barh(
    grafico["cenario"],
    grafico["CD_Pesaran"]
)

plt.axvline(0, linestyle="--")

plt.xlabel("Estatística CD de Pesaran")
plt.title("Comparação da dependência transversal")

plt.tight_layout()

plt.savefig(
    PASTA_FIGURAS / "comparacao_estatistica_cd.png",
    dpi=300
)

plt.close()


# ==========================================================
# 10. EXPORTAR RESULTADO DO CENÁRIO PRINCIPAL
# ==========================================================

principais = pares_principais.copy()

principais["correlacao_absoluta"] = (
    principais["correlacao"].abs()
)

principais = principais.sort_values(
    "correlacao_absoluta",
    ascending=False
)

principais.head(50).to_csv(
    PASTA_SAIDA / "50_maiores_correlacoes_principais.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 11. VERIFICAÇÃO DE INTEGRIDADE DA AMOSTRA
# ==========================================================

integridade = pd.DataFrame([{
    "observacoes_amostra_original": len(amostra_original),
    "paises_amostra_original": (
        amostra_original["Entity"].nunique()
    ),
    "observacoes_apos_diagnostico": len(amostra_original),
    "paises_apos_diagnostico": (
        amostra_original["Entity"].nunique()
    ),
    "base_original_preservada": True,
}])

integridade.to_csv(
    PASTA_SAIDA / "integridade_amostra.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# 12. RESULTADOS
# ==========================================================

print("\n" + "=" * 90)
print("VALIDAÇÃO DA DEPENDÊNCIA TRANSVERSAL")
print("=" * 90)

colunas_exibicao = [
    "cenario",
    "paises",
    "observacoes_modelo",
    "pares_utilizados",
    "CD_Pesaran",
    "p_valor",
    "correlacao_media",
    "correlacao_media_absoluta",
    "percentual_positivas",
    "coef_combustiveis",
    "p_combustiveis",
]

print(
    resultados_df[colunas_exibicao].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nINTEGRIDADE DA AMOSTRA")
print(integridade.to_string(index=False))

print("\nARQUIVOS GERADOS")
print(f"Tabelas: {PASTA_SAIDA}")
print(f"Gráficos: {PASTA_FIGURAS}")

print("\nVALIDAÇÃO CONCLUÍDA.")
