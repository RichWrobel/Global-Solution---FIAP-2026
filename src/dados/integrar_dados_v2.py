
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"

ARQUIVOS = {
    "acesso_eletricidade.csv": "acesso_eletricidade",
    "combustiveis_limpos.csv": "combustiveis_limpos",
    "escolaridade.csv": "escolaridade",
    "mortalidade_infantil.csv": "mortalidade_infantil",
    "pib_per_capita.csv": "pib_per_capita",
}

# Indicadores que serão integrados
dados = []

for arquivo, indicador in ARQUIVOS.items():
    caminho = RAW / arquivo

    df = pd.read_csv(caminho)

    coluna_valor = [
        c for c in df.columns
        if c not in ["Entity", "Code", "Year"]
    ]

    if len(coluna_valor) != 1:
        raise ValueError(
            f"Coluna de indicador inesperada em {arquivo}: "
            f"{coluna_valor}"
        )

    df = df[
        ["Entity", "Code", "Year", coluna_valor[0]]
    ].copy()

    df = df.rename(
        columns={coluna_valor[0]: indicador}
    )

    df["Year"] = pd.to_numeric(
        df["Year"], errors="coerce"
    )

    df[indicador] = pd.to_numeric(
        df[indicador], errors="coerce"
    )

    dados.append(df)

    print(
        f"[OK] {arquivo}: {len(df)} registros"
    )

# Integração preservando os códigos
painel = dados[0]

for df in dados[1:]:
    painel = painel.merge(
        df,
        on=["Entity", "Code", "Year"],
        how="outer",
        validate="one_to_one"
    )

painel = painel.sort_values(
    ["Entity", "Year"]
).reset_index(drop=True)

# Identificação de registros sem código de país
# Códigos ISO de países geralmente possuem três letras.
# Registros sem código serão mantidos na base completa.
painel["tipo_entidade"] = "sem_codigo"

mascara_codigo = (
    painel["Code"].notna()
    & painel["Code"].astype(str).str.fullmatch(r"[A-Z]{3}")
)

painel.loc[mascara_codigo, "tipo_entidade"] = "codigo_iso3"

# Base completa: mantém países e agregados
arquivo_completo = (
    PROCESSED / "painel_owid_completo_v2.csv"
)
painel.to_csv(
    arquivo_completo,
    index=False,
    encoding="utf-8-sig"
)

# Base inicial de modelagem: somente registros com código ISO3
painel_paises = painel[
    painel["tipo_entidade"] == "codigo_iso3"
].copy()

arquivo_paises = (
    PROCESSED / "painel_owid_paises_v2.csv"
)
painel_paises.to_csv(
    arquivo_paises,
    index=False,
    encoding="utf-8-sig"
)

# Diagnóstico de entidades
diagnostico = (
    painel.groupby("tipo_entidade")
    .agg(
        entidades=("Entity", "nunique"),
        registros=("Year", "count")
    )
    .reset_index()
)

diagnostico.to_csv(
    PROCESSED / "diagnostico_entidades_v2.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("INTEGRAÇÃO CONCLUÍDA")
print("=" * 60)

print(f"Registros na base completa: {len(painel)}")
print(f"Entidades na base completa: {painel['Entity'].nunique()}")

print(
    f"Registros na base de países: {len(painel_paises)}"
)
print(
    f"Entidades na base de países: "
    f"{painel_paises['Entity'].nunique()}"
)

print("\nDistribuição das entidades:")
print(diagnostico.to_string(index=False))

print("\nArquivos gerados:")
print(arquivo_completo)
print(arquivo_paises)
print(PROCESSED / "diagnostico_entidades_v2.csv")
