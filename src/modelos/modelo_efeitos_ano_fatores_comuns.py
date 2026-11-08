
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ============================================================
# 1. CAMINHOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]

ARQUIVO_ENTRADA = (
    RAIZ / "data" / "processed" / "painel_owid_paises_v2.csv"
)

PASTA_SAIDA = (
    RAIZ / "data" / "processed" / "modelo_efeitos_ano_fatores_comuns"
)

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARREGAMENTO E PREPARAÇÃO
# ============================================================

print("CARREGANDO DADOS...")

dados = pd.read_csv(ARQUIVO_ENTRADA)

dados = dados.sort_values(["Entity", "Year"]).copy()

variaveis = [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita",
]

colunas_obrigatorias = ["Entity", "Year"] + variaveis

ausentes = [
    coluna for coluna in colunas_obrigatorias
    if coluna not in dados.columns
]

if ausentes:
    raise ValueError(
        f"Colunas não encontradas no arquivo: {ausentes}"
    )

# Garantir valores numéricos
for coluna in variaveis:
    dados[coluna] = pd.to_numeric(
        dados[coluna], errors="coerce"
    )

dados["log_pib"] = np.where(
    dados["pib_per_capita"] > 0,
    np.log(dados["pib_per_capita"]),
    np.nan,
)

# Primeiras diferenças por país
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
        "Entity",
        "Year",
    ]
).copy()

amostra["Year"] = amostra["Year"].astype(int)

print("\nAMOSTRA PRINCIPAL")
print(f"Observações: {len(amostra)}")
print(f"Países: {amostra['Entity'].nunique()}")
print(
    f"Período: {amostra['Year'].min()} "
    f"a {amostra['Year'].max()}"
)


# ============================================================
# 3. MÉDIAS TRANSVERSAIS LEAVE-ONE-OUT
# ============================================================

print("\nCALCULANDO MÉDIAS TRANSVERSAIS...")

variaveis_diferencas = [
    "delta_mortalidade",
    "delta_combustiveis",
    "delta_log_pib",
]

for coluna in variaveis_diferencas:

    soma_anual = amostra.groupby("Year")[coluna].transform("sum")
    quantidade_anual = amostra.groupby("Year")[coluna].transform("count")

    amostra[f"{coluna}_loo"] = (
        (soma_anual - amostra[coluna])
        / (quantidade_anual - 1)
    )

    if amostra[f"{coluna}_loo"].isna().any():
        raise ValueError(
            f"Ausências encontradas na média LOO de {coluna}"
        )


# ============================================================
# 4. MODELO DE REFERÊNCIA
# ============================================================

print("\nESTIMANDO MODELO DE REFERÊNCIA...")

formula_base = (
    "delta_mortalidade ~ "
    "delta_combustiveis + delta_log_pib + C(Year)"
)

modelo_base = smf.ols(
    formula=formula_base,
    data=amostra,
).fit(
    cov_type="cluster",
    cov_kwds={"groups": amostra["Entity"]},
)


# ============================================================
# 5. MODELO COM EFEITOS DE ANO E MÉDIAS LOO
# ============================================================

print("ESTIMANDO MODELO COM FATORES COMUNS...")

formula_completo = (
    "delta_mortalidade ~ "
    "delta_combustiveis + delta_log_pib + "
    "delta_mortalidade_loo + "
    "delta_combustiveis_loo + "
    "delta_log_pib_loo + C(Year)"
)

modelo_completo = smf.ols(
    formula=formula_completo,
    data=amostra,
).fit(
    cov_type="cluster",
    cov_kwds={"groups": amostra["Entity"]},
)


# ============================================================
# 6. TESTE CONJUNTO DAS MÉDIAS TRANSVERSAIS
# ============================================================

nomes_loo = [
    "delta_mortalidade_loo",
    "delta_combustiveis_loo",
    "delta_log_pib_loo",
]

parametros = list(modelo_completo.params.index)

R = np.zeros((len(nomes_loo), len(parametros)))

for i, nome in enumerate(nomes_loo):
    if nome not in parametros:
        raise ValueError(
            f"Parâmetro não estimado: {nome}"
        )

    R[i, parametros.index(nome)] = 1

teste_loo = modelo_completo.wald_test(
    R,
    use_f=True,
    scalar=True,
)


# ============================================================
# 7. COMPARAÇÃO DOS MODELOS
# ============================================================

resultados = []

for nome, modelo in [
    ("Modelo 1 - efeitos de ano", modelo_base),
    ("Modelo 2 - efeitos de ano + médias LOO", modelo_completo),
]:

    resultados.append({
        "modelo": nome,
        "observacoes": int(modelo.nobs),
        "paises": amostra["Entity"].nunique(),
        "R2": modelo.rsquared,
        "coef_combustiveis": modelo.params["delta_combustiveis"],
        "erro_combustiveis": modelo.bse["delta_combustiveis"],
        "p_combustiveis": modelo.pvalues["delta_combustiveis"],
        "coef_log_pib": modelo.params["delta_log_pib"],
        "erro_log_pib": modelo.bse["delta_log_pib"],
        "p_log_pib": modelo.pvalues["delta_log_pib"],
    })

comparacao = pd.DataFrame(resultados)

print("\n" + "=" * 90)
print("COMPARAÇÃO DOS MODELOS")
print("=" * 90)
print(comparacao.to_string(index=False))


# ============================================================
# 8. RESULTADOS DO TESTE CONJUNTO
# ============================================================

resultado_teste = pd.DataFrame([{
    "estatistica_F": float(np.asarray(teste_loo.statistic).squeeze()),
    "p_valor": float(np.asarray(teste_loo.pvalue).squeeze()),
    "hipotese_nula": (
        "Os três coeficientes das médias transversais "
        "são conjuntamente iguais a zero"
    ),
}])

print("\nTESTE CONJUNTO DAS MÉDIAS LOO")
print(resultado_teste.to_string(index=False))


# ============================================================
# 9. DIAGNÓSTICO DE IDENTIFICAÇÃO E COLINEARIDADE
# ============================================================

matriz = modelo_completo.model.exog

diagnostico = pd.DataFrame([{
    "numero_parametros": matriz.shape[1],
    "posto_matriz": np.linalg.matrix_rank(matriz),
    "numero_condicao": np.linalg.cond(matriz),
    "matriz_com_posto_completo": (
        np.linalg.matrix_rank(matriz) == matriz.shape[1]
    ),
}])

print("\nDIAGNÓSTICO DA MATRIZ DO MODELO")
print(diagnostico.to_string(index=False))


# ============================================================
# 10. EXPORTAÇÃO
# ============================================================

comparacao.to_csv(
    PASTA_SAIDA / "comparacao_modelos.csv",
    index=False,
)

resultado_teste.to_csv(
    PASTA_SAIDA / "teste_conjunto_medias_loo.csv",
    index=False,
)

pd.DataFrame({
    "variavel": modelo_completo.params.index,
    "coeficiente": modelo_completo.params.values,
    "erro_padrao": modelo_completo.bse.values,
    "p_valor": modelo_completo.pvalues.values,
}).to_csv(
    PASTA_SAIDA / "coeficientes_modelo_completo.csv",
    index=False,
)

diagnostico.to_csv(
    PASTA_SAIDA / "diagnostico_matriz.csv",
    index=False,
)

amostra.to_csv(
    PASTA_SAIDA / "amostra_com_medias_loo.csv",
    index=False,
)

with open(
    PASTA_SAIDA / "resumo_modelo.txt",
    "w",
    encoding="utf-8",
) as arquivo:
    arquivo.write(str(modelo_completo.summary()))

print("\nARQUIVOS GERADOS:")
print(PASTA_SAIDA)

print("\nTESTE CONCLUÍDO.")
