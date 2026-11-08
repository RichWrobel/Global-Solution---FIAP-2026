
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Arquivos de origem e nomes padronizados
ARQUIVOS = {
    "acesso_eletricidade.csv": "acesso_eletricidade",
    "combustiveis_limpos.csv": "combustiveis_limpos",
    "escolaridade.csv": "escolaridade",
    "mortalidade_infantil.csv": "mortalidade_infantil",
    "pib_per_capita.csv": "pib_per_capita",
}


# ============================================================
# FUNÇÃO DE LEITURA E PADRONIZAÇÃO
# ============================================================

def carregar_base(nome_arquivo, nome_variavel):

    caminho = RAW_DIR / nome_arquivo

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    df = pd.read_csv(caminho)

    # Padronização dos nomes das colunas
    df.columns = df.columns.str.strip()

    # Verificação das colunas obrigatórias
    obrigatorias = ["Entity", "Year"]

    for coluna in obrigatorias:
        if coluna not in df.columns:
            raise ValueError(
                f"{nome_arquivo}: coluna obrigatória "
                f"'{coluna}' não encontrada."
            )

    # Identificação das colunas de indicadores
    colunas_metadados = [
        "Entity",
        "Code",
        "Year"
    ]

    colunas_indicadores = [
        coluna for coluna in df.columns
        if coluna not in colunas_metadados
    ]

    if len(colunas_indicadores) != 1:
        raise ValueError(
            f"{nome_arquivo}: esperado um indicador, "
            f"mas encontrados {colunas_indicadores}. "
            "É necessário revisar a estrutura do CSV."
        )

    coluna_original = colunas_indicadores[0]

    # Seleção e renomeação
    df = df[
        ["Entity", "Year", coluna_original]
    ].copy()

    df = df.rename(
        columns={
            coluna_original: nome_variavel
        }
    )

    # Conversão dos tipos
    df["Entity"] = df["Entity"].astype("string").str.strip()

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce"
    )

    df[nome_variavel] = pd.to_numeric(
        df[nome_variavel],
        errors="coerce"
    )

    # Remoção apenas de chaves inválidas
    df = df.dropna(
        subset=["Entity", "Year"]
    )

    df["Year"] = df["Year"].astype(int)

    # Verificação de duplicidade
    duplicados = df.duplicated(
        subset=["Entity", "Year"],
        keep=False
    )

    if duplicados.any():
        exemplos = df.loc[
            duplicados,
            ["Entity", "Year"]
        ].head(10)

        raise ValueError(
            f"Duplicidades em {nome_arquivo}:\n"
            f"{exemplos.to_string(index=False)}"
        )

    print(
        f"[OK] {nome_arquivo}: "
        f"{len(df)} registros carregados."
    )

    return df


# ============================================================
# INTEGRAÇÃO DAS BASES
# ============================================================

def integrar_bases():

    print("=" * 65)
    print("INTEGRAÇÃO DAS BASES DE DADOS")
    print("O PREÇO OCULTO DA ENERGIA SUJA")
    print("=" * 65)

    painel = None

    for arquivo, variavel in ARQUIVOS.items():

        df = carregar_base(
            arquivo,
            variavel
        )

        if painel is None:
            painel = df

        else:
            painel = painel.merge(
                df,
                on=["Entity", "Year"],
                how="outer",
                validate="one_to_one"
            )

    # Ordenação
    painel = painel.sort_values(
        ["Entity", "Year"]
    ).reset_index(drop=True)

    # ========================================================
    # DIAGNÓSTICO DE DADOS AUSENTES
    # ========================================================

    variaveis = list(ARQUIVOS.values())

    diagnostico = pd.DataFrame({
        "variavel": variaveis,
        "valores_ausentes": [
            painel[col].isna().sum()
            for col in variaveis
        ],
        "percentual_ausente": [
            round(painel[col].isna().mean() * 100, 2)
            for col in variaveis
        ]
    })

    # ========================================================
    # EXPORTAÇÃO
    # ========================================================

    arquivo_saida = (
        PROCESSED_DIR / "painel_owid_integrado.csv"
    )

    arquivo_diagnostico = (
        PROCESSED_DIR / "diagnostico_ausencias.csv"
    )

    painel.to_csv(
        arquivo_saida,
        index=False,
        encoding="utf-8-sig"
    )

    diagnostico.to_csv(
        arquivo_diagnostico,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # RELATÓRIO FINAL
    # ========================================================

    print("\n" + "=" * 65)
    print("RESULTADO DA INTEGRAÇÃO")
    print("=" * 65)

    print(f"\nTotal de registros: {len(painel)}")
    print(f"Total de entidades: {painel['Entity'].nunique()}")
    print(
        f"Período: {painel['Year'].min()} "
        f"a {painel['Year'].max()}"
    )

    print("\nDIAGNÓSTICO DE AUSÊNCIAS:")
    print(diagnostico.to_string(index=False))

    print("\nARQUIVOS GERADOS:")
    print(arquivo_saida)
    print(arquivo_diagnostico)

    print("\nIntegração concluída!")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    integrar_bases()
