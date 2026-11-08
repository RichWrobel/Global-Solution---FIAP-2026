
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE_DIR / "data" / "processed"
    / "painel_owid_paises_v2.csv"
)

SAIDA = (
    BASE_DIR / "data" / "processed"
    / "investigacao_outliers.csv"
)

df = pd.read_csv(ARQUIVO)

variaveis = [
    "mortalidade_infantil",
    "combustiveis_limpos",
    "pib_per_capita"
]

for coluna in variaveis:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

df = df[df["Year"].between(2000, 2024)].copy()
df = df.sort_values(["Entity", "Year"])

df["log_pib"] = np.where(
    df["pib_per_capita"] > 0,
    np.log(df["pib_per_capita"]),
    np.nan
)

df["delta_mortalidade"] = (
    df.groupby("Entity")["mortalidade_infantil"].diff()
)

df["delta_combustiveis"] = (
    df.groupby("Entity")["combustiveis_limpos"].diff()
)

df["delta_log_pib"] = (
    df.groupby("Entity")["log_pib"].diff()
)

df["intervalo_anual"] = df.groupby("Entity")["Year"].diff()

# Investigar os maiores movimentos da mortalidade
paises = [
    "Central African Republic",
    "Somalia",
    "South Sudan",
    "Malawi",
    "Libya",
    "Macao",
    "Indonesia",
    "Maldives",
    "Marshall Islands",
    "Guyana"
]

investigacao = df[
    df["Entity"].isin(paises)
].copy()

colunas_saida = [
    "Entity",
    "Year",
    "intervalo_anual",
    "mortalidade_infantil",
    "delta_mortalidade",
    "combustiveis_limpos",
    "delta_combustiveis",
    "pib_per_capita",
    "delta_log_pib"
]

investigacao = investigacao[colunas_saida]

investigacao.to_csv(
    SAIDA,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 70)
print("INVESTIGAÇÃO DAS SÉRIES TEMPORAIS")
print("=" * 70)

for pais in paises:
    dados = investigacao[
        investigacao["Entity"] == pais
    ].sort_values("Year")

    if not dados.empty:
        print(f"\n{'=' * 50}")
        print(pais)
        print(dados.to_string(index=False))

print("\nArquivo gerado:", SAIDA)
