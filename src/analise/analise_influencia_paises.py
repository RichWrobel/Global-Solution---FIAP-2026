
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ==========================================================
# 1. CAMINHOS E CONFIGURAÇÕES
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
# 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
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

# PIB precisa ser positivo para calcular o logaritmo.
df.loc[df["pib_per_capita"] <= 0, "pib_per_capita"] = np.nan

df["log_pib"] = np.log(df["pib_per_capita"])


# ==========================================================
# 3. CÁLCULO DAS PRIMEIRAS DIFERENÇAS
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

entidades = sorted(amostra["Entity"].unique())

print("\nDIAGNÓSTICO DA AMOSTRA")
print(f"Observações: {len(amostra)}")
print(f"Entidades: {len(entidades)}")
print(f"Período: {amostra['Year'].min()} a {amostra['Year'].max()}")


# ==========================================================
# 4. FUNÇÃO DE ESTIMAÇÃO
# ==========================================================

FORMULA = (
    "delta_mortalidade ~ "
    "delta_combustiveis + "
    "delta_log_pib + "
    "C(Year)"
)


def estimar(dados):

    modelo = smf.ols(
        formula=FORMULA,
        data=dados
    )

    resultado = modelo.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": dados["Entity"]
        }
    )

    return resultado


# ==========================================================
# 5. MODELO DE REFERÊNCIA
# ==========================================================

modelo_base = estimar(amostra)

coef_base = modelo_base.params["delta_combustiveis"]
p_base = modelo_base.pvalues["delta_combustiveis"]

print("\nMODELO ORIGINAL")
print(f"Coeficiente: {coef_base:.8f}")
print(f"p-valor: {p_base:.8f}")


# ==========================================================
# 6. LEAVE-ONE-COUNTRY-OUT
# ==========================================================

resultados = []

for i, entidade in enumerate(entidades, start=1):

    dados_reduzidos = amostra[
        amostra["Entity"] != entidade
    ].copy()

    try:
        resultado = estimar(dados_reduzidos)

        coef = resultado.params["delta_combustiveis"]
        pvalor = resultado.pvalues["delta_combustiveis"]

        resultados.append({
            "entidade_excluida": entidade,
            "observacoes": int(resultado.nobs),
            "entidades_restantes": dados_reduzidos["Entity"].nunique(),
            "coef_combustiveis": coef,
            "p_combustiveis": pvalor,
            "variacao_coeficiente": coef - coef_base,
            "variacao_absoluta": abs(coef - coef_base),
            "mudanca_sinal": np.sign(coef) != np.sign(coef_base),
            "significativo_5pct": pvalor < 0.05,
        })

    except Exception as erro:
        print(f"[ERRO] {entidade}: {erro}")

    if i % 20 == 0:
        print(f"Modelos estimados: {i}/{len(entidades)}")


# ==========================================================
# 7. ORGANIZAÇÃO DOS RESULTADOS
# ==========================================================

resultado_final = pd.DataFrame(resultados)

if resultado_final.empty:
    raise RuntimeError("Nenhum modelo LOCO foi estimado.")

resultado_final = resultado_final.sort_values(
    "variacao_absoluta",
    ascending=False
)

print("\n" + "=" * 100)
print("PAÍSES COM MAIOR INFLUÊNCIA NO COEFICIENTE")
print("=" * 100)

print(
    resultado_final.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)


# ==========================================================
# 8. RESUMO DA ESTABILIDADE
# ==========================================================

coeficientes = resultado_final["coef_combustiveis"]

mudancas_sinal = resultado_final[
    "mudanca_sinal"
].sum()

significativos = resultado_final[
    "significativo_5pct"
].sum()

resumo = pd.DataFrame([{
    "coeficiente_original": coef_base,
    "p_valor_original": p_base,
    "modelos_estimados": len(resultado_final),
    "coef_minimo": coeficientes.min(),
    "coef_maximo": coeficientes.max(),
    "coef_media": coeficientes.mean(),
    "coef_mediano": coeficientes.median(),
    "mudancas_sinal": int(mudancas_sinal),
    "modelos_significativos_5pct": int(significativos),
    "percentual_significativos": (
        significativos / len(resultado_final) * 100
    ),
}])

print("\n" + "=" * 100)
print("RESUMO DA ANÁLISE DE INFLUÊNCIA")
print("=" * 100)

print(resumo.to_string(index=False))


# ==========================================================
# 9. EXPORTAÇÃO DOS RESULTADOS
# ==========================================================

arquivo_detalhado = (
    PASTA_SAIDA / "analise_influencia_paises.csv"
)

arquivo_resumo = (
    PASTA_SAIDA / "resumo_influencia_paises.csv"
)

resultado_final.to_csv(
    arquivo_detalhado,
    index=False,
    encoding="utf-8-sig"
)

resumo.to_csv(
    arquivo_resumo,
    index=False,
    encoding="utf-8-sig"
)

print("\nARQUIVOS GERADOS:")
print(arquivo_detalhado)
print(arquivo_resumo)

print("\nANÁLISE CONCLUÍDA.")
