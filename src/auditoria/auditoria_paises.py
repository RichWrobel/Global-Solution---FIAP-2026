
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE / "data" / "processed"
    / "painel_owid_paises_v2.csv"
)

SAIDA = (
    BASE / "data" / "processed"
    / "auditoria_entidades_v2.csv"
)

df = pd.read_csv(ARQUIVO)

auditoria = (
    df.groupby(["Entity", "Code"], dropna=False)
    .agg(
        primeiro_ano=("Year", "min"),
        ultimo_ano=("Year", "max"),
        registros=("Year", "count"),
        anos_distintos=("Year", "nunique"),
        acesso_eletricidade=("acesso_eletricidade", "count"),
        combustiveis_limpos=("combustiveis_limpos", "count"),
        escolaridade=("escolaridade", "count"),
        mortalidade_infantil=("mortalidade_infantil", "count"),
        pib_per_capita=("pib_per_capita", "count"),
    )
    .reset_index()
    .sort_values("Entity")
)

auditoria.to_csv(
    SAIDA,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 65)
print("AUDITORIA DAS ENTIDADES")
print("=" * 65)

print("Total de entidades:", len(auditoria))
print("Total de códigos:", auditoria["Code"].nunique())

print("\nEntidades com maior cobertura de combustíveis limpos:")
print(
    auditoria.sort_values(
        "combustiveis_limpos", ascending=False
    ).head(15).to_string(index=False)
)

print("\nEntidades com maior cobertura de mortalidade infantil:")
print(
    auditoria.sort_values(
        "mortalidade_infantil", ascending=False
    ).head(15).to_string(index=False)
)

print("\nArquivo gerado:")
print(SAIDA)
