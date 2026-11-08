
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE / "data" / "processed"
    / "painel_owid_paises_v2.csv"
)

SAIDA = BASE / "data" / "processed"

df = pd.read_csv(ARQUIVO)

variaveis = [
    "combustiveis_limpos",
    "mortalidade_infantil",
    "pib_per_capita",
]

for coluna in variaveis:
    df[coluna] = pd.to_numeric(
        df[coluna], errors="coerce"
    )

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

df = df[df["Year"].between(2000, 2024)].copy()

# Mesma amostra do modelo principal
df = df.dropna(
    subset=[
        "combustiveis_limpos",
        "mortalidade_infantil",
        "pib_per_capita",
    ]
)

print("=" * 70)
print("DIAGNÓSTICO DE VARIAÇÃO DENTRO DOS PAÍSES")
print("=" * 70)

resultados = []

for variavel in variaveis:
    estatisticas = (
        df.groupby("Entity")[variavel]
        .agg(["count", "mean", "std", "min", "max"])
    )

    estatisticas["amplitude"] = (
        estatisticas["max"] - estatisticas["min"]
    )

    estatisticas = estatisticas.replace(
        [np.inf, -np.inf], np.nan
    )

    validos = estatisticas.dropna(subset=["std"])

    resultados.append({
        "variavel": variavel,
        "entidades": len(validos),
        "entidades_com_variacao": int(
            (validos["std"] > 0).sum()
        ),
        "entidades_sem_variacao": int(
            (validos["std"] == 0).sum()
        ),
        "desvio_padrao_medio_dentro": validos["std"].mean(),
        "amplitude_media": validos["amplitude"].mean(),
        "desvio_padrao_entre_medias": validos["mean"].std(),
    })

    print(f"\n{variavel.upper()}")
    print("Entidades:", len(validos))
    print(
        "Com variação:",
        int((validos["std"] > 0).sum())
    )
    print(
        "Sem variação:",
        int((validos["std"] == 0).sum())
    )
    print(
        "Desvio-padrão médio dentro dos países:",
        round(validos["std"].mean(), 4)
    )
    print(
        "Amplitude média:",
        round(validos["amplitude"].mean(), 4)
    )

    if variavel == "combustiveis_limpos":
        print("\n10 países com menor variação:")
        print(
            validos.sort_values("std")
            .head(10)
            .round(4)
            .to_string()
        )

        print("\n10 países com maior variação:")
        print(
            validos.sort_values("std", ascending=False)
            .head(10)
            .round(4)
            .to_string()
        )

resumo = pd.DataFrame(resultados)

resumo.to_csv(
    SAIDA / "diagnostico_variacao_paises.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("RESUMO GERAL")
print("=" * 70)
print(resumo.round(4).to_string(index=False))

print(
    "\nArquivo gerado:",
    SAIDA / "diagnostico_variacao_paises.csv"
)
