
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).resolve().parents[1]

PAINEL = (
    BASE / "data" / "processed"
    / "painel_owid_paises_v2.csv"
)

SAIDA = (
    BASE / "data" / "processed"
    / "diagnostico_residuos_ccemg"
)

SAIDA.mkdir(parents=True, exist_ok=True)

MINIMO = 20

df = pd.read_csv(PAINEL)

df = df.sort_values(["Entity", "Year"]).copy()

# Primeiras diferenças
df["delta_mortalidade"] = (
    df.groupby("Entity")["mortalidade_infantil"].diff()
)

df["delta_combustiveis"] = (
    df.groupby("Entity")["combustiveis_limpos"].diff()
)

df["delta_log_pib"] = (
    np.log(
        df["pib_per_capita"].where(
            df["pib_per_capita"] > 0
        )
    )
    .groupby(df["Entity"])
    .diff()
)

variaveis = [
    "delta_mortalidade",
    "delta_combustiveis",
    "delta_log_pib"
]

df = df.dropna(subset=variaveis).copy()

# Médias transversais anuais
medias = (
    df.groupby("Year")[variaveis]
    .mean()
    .rename(columns={
        "delta_mortalidade": "media_mortalidade",
        "delta_combustiveis": "media_combustiveis",
        "delta_log_pib": "media_log_pib"
    })
)

df = df.join(medias, on="Year")

resultados = []

for pais, grupo in df.groupby("Entity"):

    if len(grupo) < MINIMO:
        continue

    y = grupo["delta_mortalidade"]

    X = grupo[
        [
            "delta_combustiveis",
            "delta_log_pib",
            "media_mortalidade",
            "media_combustiveis",
            "media_log_pib"
        ]
    ]

    X = sm.add_constant(X)

    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        continue

    modelo = sm.OLS(y, X).fit()

    for ano, residuo in zip(
        grupo["Year"],
        modelo.resid
    ):
        resultados.append({
            "pais": pais,
            "Year": ano,
            "residuo": residuo
        })

residuos = pd.DataFrame(resultados)

if residuos.empty:
    raise ValueError(
        "Nenhum país foi estimado. Verifique os dados e o limite mínimo."
    )

residuos.to_csv(
    SAIDA / "residuos_ccemg.csv",
    index=False,
    encoding="utf-8-sig"
)

# Matriz de resíduos: países nas linhas, anos nas colunas
matriz = residuos.pivot(
    index="pais",
    columns="Year",
    values="residuo"
)

correlacoes = matriz.T.corr(min_periods=10)


# Extrair correlações únicas, sem diagonal e sem duplicar pares
pares = []

paises = correlacoes.index.tolist()

for i in range(len(paises)):
    for j in range(i + 1, len(paises)):

        valor = correlacoes.iloc[i, j]

        if pd.notna(valor):
            pares.append({
                "pais_1": paises[i],
                "pais_2": paises[j],
                "correlacao": valor
            })

pares_df = pd.DataFrame(pares)

valores = pares_df["correlacao"]

pares_df.to_csv(
    SAIDA / "pares_correlacao_residuos.csv",
    index=False,
    encoding="utf-8-sig"
)

resumo = pd.DataFrame([{
    "paises": matriz.shape[0],
    "anos": matriz.shape[1],
    "pares": len(valores),
    "correlacao_media": valores.mean(),
    "correlacao_mediana": valores.median(),
    "correlacao_media_absoluta": valores.abs().mean(),
    "percentual_correlacoes_positivas": (
        (valores > 0).mean() * 100
    ),
    "correlacao_minima": valores.min(),
    "correlacao_maxima": valores.max()
}])

resumo.to_csv(
    SAIDA / "resumo_dependencia_residuos.csv",
    index=False,
    encoding="utf-8-sig"
)

correlacoes.to_csv(
    SAIDA / "matriz_correlacao_residuos.csv",
    encoding="utf-8-sig"
)

print("=" * 80)
print("DIAGNÓSTICO DESCRITIVO DOS RESÍDUOS CCEMG")
print("=" * 80)

print("\nPaíses estimados:", matriz.shape[0])
print("Anos disponíveis:", matriz.shape[1])
print("\nRESUMO DAS CORRELAÇÕES ENTRE RESÍDUOS")

print(resumo.round(6).to_string(index=False))


print("\n10 MAIORES CORRELAÇÕES POSITIVAS")

positivas = pares_df.loc[
    pares_df["correlacao"] > 0
].sort_values(
    "correlacao",
    ascending=False
).head(10)

print(positivas.to_string(index=False))

print("\n10 MENORES CORRELAÇÕES NEGATIVAS")

negativas = pares_df.loc[
    pares_df["correlacao"] < 0
].sort_values(
    "correlacao",
    ascending=True
).head(10)

print(negativas.to_string(index=False))

print("\nTESTE DE INTEGRIDADE")

print("Colunas:", pares_df.columns.tolist())
print("Total de pares:", len(pares_df))
print("Pares positivos:", (pares_df["correlacao"] > 0).sum())
print("Pares negativos:", (pares_df["correlacao"] < 0).sum())

print("\nDIAGNÓSTICO CONCLUÍDO.")
