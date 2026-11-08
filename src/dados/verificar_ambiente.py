
import sys
import importlib

print("=" * 50)
print("VERIFICAÇÃO DO AMBIENTE")
print("=" * 50)

print(f"\nPython: {sys.version}")
print(f"Executável: {sys.executable}")

bibliotecas = [
    "pandas",
    "numpy",
    "streamlit",
    "plotly",
    "sklearn",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "requests"
]

print("\nVERIFICAÇÃO DAS BIBLIOTECAS")

for biblioteca in bibliotecas:
    try:
        modulo = importlib.import_module(biblioteca)
        versao = getattr(modulo, "__version__", "Versão não informada")
        print(f"[OK] {biblioteca}: {versao}")
    except ImportError:
        print(f"[AUSENTE] {biblioteca}")

print("\nVerificação concluída.")
