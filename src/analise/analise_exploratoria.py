
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "painel_owid_integrado.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "eda"
FIGURES_DIR = BASE_DIR / "figuras" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INDICADORES = [
    "acesso_eletricidade",
    "combustiveis_limpos",
    "escolaridade",
    "mortalidade_infantil",
    "pib_per_capita"
]

sns.set_theme(style="whitegrid")


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_dados():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Base integrada não encontrada: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Base carregada: {df.shape[0]} registros")
    print(f"Entidades: {df['Entity'].nunique()}")

    return df


# ============================================================
# 1. COBERTURA DOS INDICADORES
# ============================================================

def analisar_cobertura(df):

    resultados = []

    for indicador in INDICADORES:

        dados = df[indicador].dropna()

        resultados.append({
            "indicador": indicador,
            "observacoes_validas": len(dados),
            "observacoes_ausentes": df[indicador].isna().sum(),
            "cobertura_percentual": round(
                df[indicador].notna().mean() * 100, 2
            ),
            "valor_minimo": dados.min() if len(dados) else None,
            "valor_maximo": dados.max() if len(dados) else None,
            "media": dados.mean() if len(dados) else None,
            "mediana": dados.median() if len(dados) else None
        })

    resultado = pd.DataFrame(resultados)

    resultado.to_csv(
        OUTPUT_DIR / "cobertura_indicadores.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCOBERTURA DOS INDICADORES")
    print(resultado.to_string(index=False))

    return resultado


# ============================================================
# 2. COBERTURA POR ANO
# ============================================================

def analisar_cobertura_temporal(df):

    cobertura = (
        df.groupby("Year")[INDICADORES]
        .count()
        .reset_index()
    )

    cobertura.to_csv(
        OUTPUT_DIR / "cobertura_por_ano.csv",
        index=False,
        encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    for indicador in INDICADORES:

        ax.plot(
            cobertura["Year"],
            cobertura[indicador],
            label=indicador,
            linewidth=2
        )

    ax.set_title(
        "Disponibilidade histórica dos indicadores"
    )

    ax.set_xlabel("Ano")
    ax.set_ylabel("Quantidade de observações")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "cobertura_temporal.png",
        dpi=300
    )

    plt.close(fig)

    print("\nCobertura temporal calculada.")


# ============================================================
# 3. COBERTURA POR ENTIDADE
# ============================================================

def analisar_cobertura_entidades(df):

    cobertura = df.groupby("Entity").agg(
        anos_disponiveis=("Year", "nunique"),
        acesso_eletricidade=("acesso_eletricidade", "count"),
        combustiveis_limpos=("combustiveis_limpos", "count"),
        escolaridade=("escolaridade", "count"),
        mortalidade_infantil=("mortalidade_infantil", "count"),
        pib_per_capita=("pib_per_capita", "count")
    ).reset_index()

    cobertura["indicadores_com_dados"] = (
        df.groupby("Entity")[INDICADORES]
        .count()
        .gt(0)
        .sum(axis=1)
        .reindex(cobertura["Entity"])
        .values
    )

    cobertura.to_csv(
        OUTPUT_DIR / "cobertura_entidades.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nENTIDADES COM MAIOR DISPONIBILIDADE")

    print(
        cobertura.sort_values(
            "anos_disponiveis",
            ascending=False
        ).head(15).to_string(index=False)
    )


# ============================================================
# 4. CORRELAÇÕES
# ============================================================

def analisar_correlacoes(df):

    correlacao = df[INDICADORES].corr(
        method="pearson"
    )

    correlacao.to_csv(
        OUTPUT_DIR / "matriz_correlacao.csv",
        encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        correlacao,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".2f",
        ax=ax
    )

    ax.set_title(
        "Correlação entre os indicadores"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "matriz_correlacao.png",
        dpi=300
    )

    plt.close(fig)

    print("\nMATRIZ DE CORRELAÇÃO")
    print(correlacao.round(3))


# ============================================================
# 5. AMOSTRA PARA MODELAGEM
# ============================================================

def analisar_amostras(df):

    combinacoes = [
        ["acesso_eletricidade", "mortalidade_infantil"],
        ["combustiveis_limpos", "mortalidade_infantil"],
        [
            "combustiveis_limpos",
            "mortalidade_infantil",
            "pib_per_capita"
        ],
        [
            "combustiveis_limpos",
            "mortalidade_infantil",
            "pib_per_capita",
            "escolaridade"
        ]
    ]

    resultados = []

    for variaveis in combinacoes:

        amostra = df.dropna(subset=variaveis)

        resultados.append({
            "variaveis": " + ".join(variaveis),
            "observacoes": len(amostra),
            "entidades": amostra["Entity"].nunique(),
            "anos": amostra["Year"].nunique()
        })

    resultado = pd.DataFrame(resultados)

    resultado.to_csv(
        OUTPUT_DIR / "amostras_modelagem.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAMOSTRAS DISPONÍVEIS PARA MODELAGEM")
    print(resultado.to_string(index=False))


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    print("=" * 65)
    print("ANÁLISE EXPLORATÓRIA DOS DADOS")
    print("=" * 65)

    df = carregar_dados()

    analisar_cobertura(df)
    analisar_cobertura_temporal(df)
    analisar_cobertura_entidades(df)
    analisar_correlacoes(df)
    analisar_amostras(df)

    print("\n" + "=" * 65)
    print("ANÁLISE EXPLORATÓRIA CONCLUÍDA")
    print("=" * 65)

    print(f"\nResultados: {OUTPUT_DIR}")
    print(f"Gráficos: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
