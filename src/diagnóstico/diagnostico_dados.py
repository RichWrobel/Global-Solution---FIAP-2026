
from pathlib import Path
import pandas as pd

# Caminho principal do projeto
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

print("=" * 70)
print("DIAGNÓSTICO DAS BASES DE DADOS")
print("PROJETO: O PREÇO OCULTO DA ENERGIA SUJA")
print("=" * 70)

if not RAW_DIR.exists():
    raise FileNotFoundError(
        f"Pasta de dados não encontrada: {RAW_DIR}"
    )

arquivos = sorted(RAW_DIR.glob("*.csv"))

if not arquivos:
    raise FileNotFoundError(
        "Nenhum arquivo CSV encontrado em data/raw."
    )

for arquivo in arquivos:

    print("\n" + "=" * 70)
    print(f"ARQUIVO: {arquivo.name}")
    print("=" * 70)

    try:
        df = pd.read_csv(arquivo)

        print(f"\nQuantidade de linhas: {df.shape[0]}")
        print(f"Quantidade de colunas: {df.shape[1]}")

        print("\nCOLUNAS:")
        for coluna in df.columns:
            print(f"- {coluna}")

        print("\nTIPOS DE DADOS:")
        print(df.dtypes.to_string())

        print("\nVALORES AUSENTES:")
        ausentes = df.isna().sum()
        percentual = (df.isna().mean() * 100).round(2)

        diagnostico = pd.DataFrame({
            "Valores ausentes": ausentes,
            "Percentual (%)": percentual
        })

        print(diagnostico.to_string())

        print("\nPRIMEIRAS CINCO OBSERVAÇÕES:")
        print(df.head().to_string(index=False))

        print("\nPERÍODO DOS DADOS:")

        if "Year" in df.columns:
            print(
                f"Ano inicial: {df['Year'].min()}"
            )
            print(
                f"Ano final: {df['Year'].max()}"
            )

        print("\nPAÍSES OU ENTIDADES:")
        if "Entity" in df.columns:
            print(
                f"Quantidade de entidades: "
                f"{df['Entity'].nunique()}"
            )

    except Exception as erro:
        print(f"ERRO AO LER O ARQUIVO: {erro}")

print("\n" + "=" * 70)
print("DIAGNÓSTICO CONCLUÍDO")
print("=" * 70)
