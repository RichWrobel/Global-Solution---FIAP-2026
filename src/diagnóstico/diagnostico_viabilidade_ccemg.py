
from pathlib import Path
import pandas as pd
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]

ENTRADA = (
    RAIZ / "data" / "processed" /
    "painel_owid_paises_v2.csv"
)

SAIDA = (
    RAIZ / "data" / "processed" /
    "diagnostico_viabilidade_ccemg"
)

SAIDA.mkdir(parents=True, exist_ok=True)

dados = pd.read_csv(ENTRADA)
dados = dados.sort_values(["Entity", "Year"]).copy()

variaveis = [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita",
]

for coluna in variaveis:
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

# Médias transversais anuais, calculadas na amostra comum.
variaveis_modelo = [
    "delta_mortalidade",
    "delta_combustiveis",
    "delta_log_pib",
]

medias = (
    amostra.groupby("Year")[variaveis_modelo]
    .mean()
    .add_suffix("_media")
)

amostra = amostra.merge(
    medias,
    on="Year",
    how="left",
    validate="many_to_one",
)

resultados = []

for pais, grupo in amostra.groupby("Entity"):

    grupo = grupo.sort_values("Year")

    n = len(grupo)
    anos = grupo["Year"].nunique()

    # CCEMG em primeiras diferenças:
    # 2 regressores individuais + 3 médias transversais
    # + constante.
    numero_parametros = 6

    resultados.append({
        "pais": pais,
        "observacoes": n,
        "anos_distintos": anos,
        "observacoes_suficientes_10": n >= 10,
        "observacoes_suficientes_15": n >= 15,
        "observacoes_suficientes_20": n >= 20,
        "graus_liberdade_residuais": n - numero_parametros,
    })

diagnostico = pd.DataFrame(resultados)

print("\nDIAGNÓSTICO DE VIABILIDADE CCEMG")
print(f"Países: {len(diagnostico)}")
print(f"Observações totais: {len(amostra)}")

print("\nDISTRIBUIÇÃO DAS OBSERVAÇÕES POR PAÍS")
print(diagnostico["observacoes"].describe().to_string())

print("\nPAÍSES POR LIMIAR DE OBSERVAÇÕES")
print(
    pd.DataFrame({
        "criterio": [10, 15, 20],
        "paises": [
            diagnostico["observacoes_suficientes_10"].sum(),
            diagnostico["observacoes_suficientes_15"].sum(),
            diagnostico["observacoes_suficientes_20"].sum(),
        ],
    }).to_string(index=False)
)

print("\nPAÍSES COM MENOS DE 10 OBSERVAÇÕES")
print(
    diagnostico.loc[
        diagnostico["observacoes"] < 10
    ].to_string(index=False)
)

print("\nPAÍSES COM 20 OU MAIS OBSERVAÇÕES")
print(
    int((diagnostico["observacoes"] >= 20).sum())
)

diagnostico.to_csv(
    SAIDA / "observacoes_por_pais.csv",
    index=False,
)

print("\nARQUIVO SALVO EM:")
print(SAIDA)

print("\nDIAGNÓSTICO CONCLUÍDO.")
