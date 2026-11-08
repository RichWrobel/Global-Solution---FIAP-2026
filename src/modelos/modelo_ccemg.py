
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ============================================================
# 1. CAMINHOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]

ENTRADA = (
    RAIZ / "data" / "processed" /
    "painel_owid_paises_v2.csv"
)

SAIDA = (
    RAIZ / "data" / "processed" /
    "modelo_ccemg"
)

SAIDA.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. PREPARAÇÃO DOS DADOS
# ============================================================

print("CARREGANDO DADOS...")

dados = pd.read_csv(ENTRADA)

dados = dados.sort_values(
    ["Entity", "Year"]
).copy()

colunas = [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita",
]

for coluna in colunas:
    dados[coluna] = pd.to_numeric(
        dados[coluna], errors="coerce"
    )

dados["log_pib"] = np.where(
    dados["pib_per_capita"] > 0,
    np.log(dados["pib_per_capita"]),
    np.nan,
)

dados["delta_mortalidade"] = (
    dados.groupby("Entity")["mortalidade_infantil"].diff()
)

dados["delta_combustiveis"] = (
    dados.groupby("Entity")["combustiveis_limpos"].diff()
)

dados["delta_log_pib"] = (
    dados.groupby("Entity")["log_pib"].diff()
)

amostra = dados.dropna(
    subset=[
        "delta_mortalidade",
        "delta_combustiveis",
        "delta_log_pib",
    ]
).copy()

amostra["Year"] = amostra["Year"].astype(int)

print("\nAMOSTRA")
print(f"Observações: {len(amostra)}")
print(f"Países: {amostra['Entity'].nunique()}")
print(
    f"Período: {amostra['Year'].min()} "
    f"a {amostra['Year'].max()}"
)


# ============================================================
# 3. MÉDIAS TRANSVERSAIS
# ============================================================

print("\nCALCULANDO MÉDIAS TRANSVERSAIS...")

variaveis = [
    "delta_mortalidade",
    "delta_combustiveis",
    "delta_log_pib",
]

medias = (
    amostra.groupby("Year")[variaveis]
    .mean()
    .add_suffix("_media")
)

amostra = amostra.merge(
    medias,
    on="Year",
    how="left",
    validate="many_to_one",
)

regressores_base = [
    "delta_combustiveis",
    "delta_log_pib",
]

regressores_cce = [
    "delta_mortalidade_media",
    "delta_combustiveis_media",
    "delta_log_pib_media",
]

resultados_pais = []
resultados_modelos = []


# ============================================================
# 4. FUNÇÃO DE ESTIMAÇÃO MEAN GROUP
# ============================================================

def estimar_mean_group(base, minimo, incluir_medias):

    nome_modelo = (
        "CCEMG" if incluir_medias else "MG"
    )

    coeficientes = []

    diagnosticos = []

    for pais, grupo in base.groupby("Entity"):

        grupo = grupo.sort_values("Year").copy()

        if len(grupo) < minimo:
            continue

        X_colunas = regressores_base.copy()

        if incluir_medias:
            X_colunas += regressores_cce

        X = sm.add_constant(
            grupo[X_colunas].astype(float),
            has_constant="add",
        )

        y = grupo["delta_mortalidade"].astype(float)

        matriz = X.to_numpy()

        posto = np.linalg.matrix_rank(matriz)

        if posto < matriz.shape[1]:
            diagnosticos.append({
                "pais": pais,
                "observacoes": len(grupo),
                "modelo": nome_modelo,
                "situacao": "Matriz sem posto completo",
                "posto": posto,
                "parametros": matriz.shape[1],
                "numero_condicao": np.linalg.cond(matriz),
            })
            continue

        if len(grupo) <= matriz.shape[1]:
            continue

        modelo = sm.OLS(y, X).fit()

        numero_condicao = np.linalg.cond(matriz)

        registro = {
            "pais": pais,
            "observacoes": len(grupo),
            "modelo": nome_modelo,
            "situacao": "Estimado",
            "posto": posto,
            "parametros": matriz.shape[1],
            "numero_condicao": numero_condicao,
        }

        diagnosticos.append(registro)

        coeficientes.append({
            "pais": pais,
            "observacoes": len(grupo),
            "coef_combustiveis": modelo.params[
                "delta_combustiveis"
            ],
            "coef_log_pib": modelo.params[
                "delta_log_pib"
            ],
        })

    coef_df = pd.DataFrame(coeficientes)

    if coef_df.empty:
        return None, coef_df, diagnosticos

    n = len(coef_df)

    linhas = []

    for variavel, coluna in [
        ("Combustíveis limpos", "coef_combustiveis"),
        ("Logaritmo do PIB", "coef_log_pib"),
    ]:

        valores = coef_df[coluna]

        media = valores.mean()
        desvio = valores.std(ddof=1)
        erro_padrao = desvio / np.sqrt(n)

        estatistica_t = media / erro_padrao

        p_valor = 2 * stats.t.sf(
            abs(estatistica_t),
            df=n - 1,
        )

        critico = stats.t.ppf(
            0.975,
            df=n - 1,
        )

        linhas.append({
            "modelo": nome_modelo,
            "limite_minimo_observacoes": minimo,
            "variavel": variavel,
            "paises_estimados": n,
            "coeficiente_medio": media,
            "desvio_padrao_coeficientes": desvio,
            "erro_padrao_mean_group": erro_padrao,
            "estatistica_t": estatistica_t,
            "p_valor": p_valor,
            "ic95_inferior": media - critico * erro_padrao,
            "ic95_superior": media + critico * erro_padrao,
        })

    return pd.DataFrame(linhas), coef_df, diagnosticos


# ============================================================
# 5. ESTIMAÇÕES E SENSIBILIDADE
# ============================================================

for minimo in [10, 15, 20]:

    print(f"\nESTIMANDO MODELOS COM MÍNIMO DE {minimo} OBSERVAÇÕES")

    for incluir_medias in [False, True]:

        resultado, coeficientes, diagnosticos = (
            estimar_mean_group(
                amostra,
                minimo,
                incluir_medias,
            )
        )

        if resultado is not None:
            resultados_modelos.append(resultado)

            coeficientes["limite_minimo_observacoes"] = minimo

            coeficientes["modelo"] = (
                "CCEMG" if incluir_medias else "MG"
            )

            resultados_pais.append(coeficientes)

        pd.DataFrame(diagnosticos).to_csv(
            SAIDA /
            f"diagnostico_{'CCEMG' if incluir_medias else 'MG'}_minimo_{minimo}.csv",
            index=False,
        )


# ============================================================
# 6. RESULTADOS
# ============================================================

comparacao = pd.concat(
    resultados_modelos,
    ignore_index=True,
)

coeficientes_pais = pd.concat(
    resultados_pais,
    ignore_index=True,
)

print("\n" + "=" * 100)
print("RESULTADOS MEAN GROUP E CCEMG")
print("=" * 100)

print(
    comparacao.to_string(index=False)
)


# ============================================================
# 7. EXPORTAÇÃO
# ============================================================

comparacao.to_csv(
    SAIDA / "comparacao_mean_group_ccemg.csv",
    index=False,
)

coeficientes_pais.to_csv(
    SAIDA / "coeficientes_por_pais.csv",
    index=False,
)

amostra.to_csv(
    SAIDA / "amostra_com_medias_transversais.csv",
    index=False,
)

print("\nARQUIVOS GERADOS EM:")
print(SAIDA)

print("\nESTIMAÇÃO CONCLUÍDA.")
