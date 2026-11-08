
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE
    / "data"
    / "processed"
    / "modelo_ccemg"
    / "coeficientes_por_pais.csv"
)

SAIDA = (
    BASE
    / "data"
    / "processed"
    / "sensibilidade_ccemg"
)

SAIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ARQUIVO)

variaveis = [
    "coef_combustiveis",
    "coef_log_pib"
]

print("=" * 80)
print("SENSIBILIDADE DOS COEFICIENTES CCEMG")
print("=" * 80)

resultados = []

for variavel in variaveis:

    dados = df[["pais", variavel]].dropna().copy()

    dados[variavel] = pd.to_numeric(
        dados[variavel],
        errors="coerce"
    )

    dados = dados.dropna()

    cenarios = {
        "Original": dados,
        "Sem 1% extremos": dados[
            (dados[variavel] >= dados[variavel].quantile(0.01))
            & (dados[variavel] <= dados[variavel].quantile(0.99))
        ],
        "Sem 5% extremos": dados[
            (dados[variavel] >= dados[variavel].quantile(0.05))
            & (dados[variavel] <= dados[variavel].quantile(0.95))
        ],
        "Sem 10% extremos": dados[
            (dados[variavel] >= dados[variavel].quantile(0.10))
            & (dados[variavel] <= dados[variavel].quantile(0.90))
        ]
    }

    for nome, amostra in cenarios.items():

        coeficientes = amostra[variavel]

        resultados.append({
            "variavel": variavel,
            "cenario": nome,
            "paises": len(coeficientes),
            "media": coeficientes.mean(),
            "mediana": coeficientes.median(),
            "desvio_padrao": coeficientes.std(),
            "minimo": coeficientes.min(),
            "maximo": coeficientes.max()
        })

resultado = pd.DataFrame(resultados)

resultado.to_csv(
    SAIDA / "sensibilidade_medias_ccemg.csv",
    index=False,
    encoding="utf-8-sig"
)

for variavel in variaveis:

    print(f"\n\nVARIÁVEL: {variavel}")

    tabela = resultado[
        resultado["variavel"] == variavel
    ]

    print(
        tabela[
            [
                "cenario",
                "paises",
                "media",
                "mediana",
                "desvio_padrao",
                "minimo",
                "maximo"
            ]
        ].round(6).to_string(index=False)
    )

print("\n\nARQUIVO GERADO:")
print(SAIDA / "sensibilidade_medias_ccemg.csv")

print("\nANÁLISE CONCLUÍDA.")
