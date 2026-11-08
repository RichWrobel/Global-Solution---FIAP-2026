
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE / "data" / "processed"
    / "painel_owid_paises_v2.csv"
)

SAIDA = BASE / "data" / "processed"

df = pd.read_csv(ARQUIVO)

indicadores = [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita",
    "escolaridade",
]

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Período de interesse
df = df[
    df["Year"].between(2000, 2024)
].copy()

for coluna in indicadores:
    df[coluna] = pd.to_numeric(
        df[coluna], errors="coerce"
    )

# Amostra principal
principal = df.dropna(
    subset=[
        "mortalidade_infantil",
        "combustiveis_limpos",
        "pib_per_capita",
    ]
).copy()

# Amostra ampliada
ampliada = principal.dropna(
    subset=["escolaridade"]
).copy()

def resumo(nome, base):
    print("\n" + "=" * 65)
    print(nome)
    print("=" * 65)

    print("Observações:", len(base))
    print("Entidades:", base["Entity"].nunique())

    if len(base):
        print(
            "Período:",
            int(base["Year"].min()),
            "a",
            int(base["Year"].max())
        )

        print(
            "Anos distintos:",
            base["Year"].nunique()
        )

        print(
            "Observações por entidade:",
            round(
                len(base) / base["Entity"].nunique(),
                2
            )
        )

        print(
            "Entidades por ano (média):",
            round(
                base.groupby("Year")["Entity"].nunique().mean(),
                2
            )
        )

        print(
            "Menor cobertura anual:",
            base.groupby("Year")["Entity"].nunique().min()
        )

        print(
            "Maior cobertura anual:",
            base.groupby("Year")["Entity"].nunique().max()
        )

resumo("AMOSTRA PRINCIPAL", principal)
resumo("AMOSTRA COM ESCOLARIDADE", ampliada)

cobertura = (
    principal.groupby("Year")
    .agg(
        observacoes=("Entity", "count"),
        entidades=("Entity", "nunique")
    )
    .reset_index()
)

cobertura.to_csv(
    SAIDA / "cobertura_amostra_principal.csv",
    index=False,
    encoding="utf-8-sig"
)

comparacao = pd.DataFrame([
    {
        "amostra": "Principal",
        "observacoes": len(principal),
        "entidades": principal["Entity"].nunique(),
        "anos": principal["Year"].nunique(),
    },
    {
        "amostra": "Com escolaridade",
        "observacoes": len(ampliada),
        "entidades": ampliada["Entity"].nunique(),
        "anos": ampliada["Year"].nunique(),
    }
])

comparacao.to_csv(
    SAIDA / "comparacao_amostras.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArquivos gerados:")
print(SAIDA / "cobertura_amostra_principal.csv")
print(SAIDA / "comparacao_amostras.csv")
