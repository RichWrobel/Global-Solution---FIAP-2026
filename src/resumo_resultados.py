
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
EDA = BASE / "data" / "processed" / "eda"

ARQUIVOS = [
    "amostras_modelagem.csv",
    "cobertura_indicadores.csv",
    "cobertura_entidades.csv",
    "matriz_correlacao.csv",
    "diagnostico_ausencias.csv",
]

for nome in ARQUIVOS:
    caminho = EDA / nome

    print("\n" + "=" * 70)
    print(f"ARQUIVO: {nome}")
    print("=" * 70)

    if not caminho.exists():
        print(f"[NÃO ENCONTRADO] {caminho}")
        continue

    try:
        df = pd.read_csv(caminho)

        print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print("\nColunas:")
        print(list(df.columns))

        print("\nConteúdo:")
        print(df.head(10).to_string(index=False))

    except Exception as erro:
        print(f"[ERRO] {erro}")

print("\nRelatório concluído.")
