from pathlib import Path
import pandas as pd
import py_compile

ROOT = Path.cwd()
print("\n" + "="*60)
print("AUDITORIA FINAL — FIAP GLOBAL SOLUTION 2026.2")
print("="*60)

# 1. Código Python
arquivos_py = [
    p for p in ROOT.rglob("*.py")
    if ".venv" not in p.parts and "__pycache__" not in p.parts
]
erros = []
for p in arquivos_py:
    try:
        py_compile.compile(str(p), doraise=True)
    except Exception as e:
        erros.append((str(p.relative_to(ROOT)), str(e)))

print(f"\n[1] CÓDIGO: {len(arquivos_py)} arquivos Python")
print("ERROS:", len(erros))
for item in erros:
    print(" -", item)

# 2. Arquivos e dados essenciais
essenciais = [
    "src/app.py",
    "data/processed/painel_owid_paises_v2.csv",
    "data/processed/painel_clusters.csv",
    "data/processed/ranking_vulnerabilidade_2020.csv",
    "data/processed/lstm_backtest_recursivo_2020_2024.csv",
    "data/processed/lstm_backtest_metricas_anuais.csv",
    "data/processed/lstm_sensibilidade.csv",
    "data/processed/lstm_previsoes_2025_2029.csv",
    "models/lstm_global.keras",
    "requirements.txt",
]
faltantes = [f for f in essenciais if not (ROOT / f).exists()]
print(f"\n[2] ARQUIVOS: {len(essenciais)-len(faltantes)}/{len(essenciais)} encontrados")
for f in faltantes:
    print(" - FALTA:", f)

# 3. Integridade e período dos dados
print("\n[3] DADOS")
for nome in [
    "painel_owid_paises_v2.csv",
    "painel_clusters.csv",
    "ranking_vulnerabilidade_2020.csv",
    "lstm_backtest_recursivo_2020_2024.csv",
    "lstm_backtest_metricas_anuais.csv",
    "lstm_sensibilidade.csv",
    "lstm_previsoes_2025_2029.csv",
]:
    p = ROOT / "data" / "processed" / nome
    if p.exists():
        try:
            df = pd.read_csv(p)
            print(f" - {nome}: {len(df):,} linhas | {len(df.columns)} colunas")
            if "Year" in df.columns:
                print(f"   Período: {df['Year'].min()}–{df['Year'].max()}")
        except Exception as e:
            print(f" - ERRO ao ler {nome}: {e}")

# 4. Resultados do backtest recursivo
p = ROOT / "data/processed/lstm_backtest_metricas.json"
if p.exists():
    import json
    m = json.loads(p.read_text(encoding="utf-8"))
    print("\n[4] BACKTEST LSTM RECURSIVO")
    for k, v in m.items():
        print(f" - {k}: {v}")

# 5. Dashboard e rótulos
app = ROOT / "src/app.py"
codigo = app.read_text(encoding="utf-8", errors="replace") if app.exists() else ""
print("\n[5] DASHBOARD")
for termo in ["Previsão LSTM", "lstm_previsoes_2025_2029.csv",
              "lstm_backtest_recursivo_2020_2024.csv"]:
    print(f" - {termo}: {'OK' if termo in codigo else 'NÃO ENCONTRADO'}")

grafico = ROOT / "src/FrontEnd/grafico.py"
g = grafico.read_text(encoding="utf-8", errors="replace") if grafico.exists() else ""
rotulo_antigo = "Mortalidade infantil (por 1.000)"
print(" - Rótulo antigo de mortalidade:",
      "CORRIGIR" if rotulo_antigo in g else "não encontrado")

# 6. Possíveis arquivos sensíveis
print("\n[6] SEGURANÇA")
sensíveis = [
    p for p in ROOT.rglob("*")
    if p.is_file()
    and ".venv" not in p.parts
    and any(x in p.name.lower() for x in [".env", "secret", "credential", "token"])
]
print(" - Arquivos com nomes potencialmente sensíveis:", len(sensíveis))
for p in sensíveis:
    print("   ", p.relative_to(ROOT))

print("\n" + "="*60)
print("FIM DA AUDITORIA")
print("="*60)
