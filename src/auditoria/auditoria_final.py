from pathlib import Path
import sys
import subprocess
import importlib
import pandas as pd

ROOT = Path.cwd()
linhas = []

def registrar(titulo, conteudo):
    linhas.append("\n" + "=" * 75)
    linhas.append(titulo)
    linhas.append("=" * 75)
    linhas.append(str(conteudo))

def verificar_arquivo(caminho):
    p = ROOT / caminho
    return f"{'OK' if p.exists() else 'FALTA'} | {caminho}"

print("\nINICIANDO AUDITORIA FINAL DO PROJETO...\n")

# 1. Estrutura principal
arquivos = [
    "requirements.txt",
    "src/app.py",
    "src/modelo_ccemg.py",
    "src/modelo_efeitos_fixos_v3.py",
    "src/FrontEnd/componentes.py",
    "src/FrontEnd/estilos.py",
    "src/FrontEnd/grafico.py",
    "data/raw/combustiveis_limpos.csv",
    "data/raw/mortalidade_infantil.csv",
    "data/processed/painel_owid_paises_v2.csv",
    "data/processed/painel_clusters.csv",
    "data/processed/ranking_vulnerabilidade_2020.csv",
    "data/processed/comparacao_modelos_v3.csv",
    "data/processed/modelo_ccemg/comparacao_mean_group_ccemg.csv",
    "data/processed/modelo_ccemg/coeficientes_por_pais.csv",
    "data/processed/modelo_ccemg/bootstrap_coeficientes_por_pais.csv",
    "data/processed/modelo_ccemg/auditoria_influencia_haiti_somalia.csv",
    "figuras/previsao_lstm.png",
]

registrar("1. VERIFICAÇÃO DE ARQUIVOS",
          "\n".join(verificar_arquivo(a) for a in arquivos))

# 2. Bibliotecas
bibliotecas = [
    "streamlit", "plotly", "pandas", "numpy", "sklearn",
    "statsmodels", "scipy", "matplotlib", "seaborn", "requests"
]
resultado_bibliotecas = []
for nome in bibliotecas:
    try:
        importlib.import_module(nome)
        resultado_bibliotecas.append(f"OK | {nome}")
    except Exception as e:
        resultado_bibliotecas.append(f"ERRO | {nome}: {e}")

registrar("2. BIBLIOTECAS PYTHON",
          "\n".join(resultado_bibliotecas))

# 3. Compilação dos códigos
src = ROOT / "src"
if src.exists():
    arquivos_py = list(src.rglob("*.py"))
    compilacao = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", str(src)],
        capture_output=True, text=True
    )
    texto = f"Arquivos Python encontrados: {len(arquivos_py)}\n"
    texto += f"Resultado da compilação: {'OK' if compilacao.returncode == 0 else 'ERRO'}\n"
    if compilacao.stderr:
        texto += compilacao.stderr
else:
    texto = "Pasta src não encontrada."

registrar("3. COMPILAÇÃO DOS CÓDIGOS", texto)

# 4. Integridade das bases
bases = [
    "data/processed/painel_owid_paises_v2.csv",
    "data/processed/painel_clusters.csv",
    "data/processed/ranking_vulnerabilidade_2020.csv",
    "data/processed/comparacao_modelos_v3.csv",
    "data/processed/modelo_ccemg/comparacao_mean_group_ccemg.csv",
    "data/processed/modelo_ccemg/bootstrap_coeficientes_por_pais.csv",
]

for arquivo in bases:
    p = ROOT / arquivo
    if p.exists():
        try:
            df = pd.read_csv(p)
            linhas.append(
                f"\n{arquivo}\n"
                f"Dimensões: {df.shape}\n"
                f"Colunas: {df.columns.tolist()}"
            )
        except Exception as e:
            linhas.append(f"\nERRO NA LEITURA: {arquivo} | {e}")

registrar("4. INTEGRIDADE E DIMENSÕES DAS BASES", "\n".join(linhas[-len(bases):]))

# 5. Perfis dos agrupamentos
p = ROOT / "data/processed/painel_clusters.csv"
if p.exists():
    df = pd.read_csv(p)
    texto = df["perfil"].value_counts().to_string() if "perfil" in df else "Coluna perfil ausente."
    registrar("5. PERFIS DOS PAÍSES E TERRITÓRIOS", texto)

# 6. Resultados econométricos
p = ROOT / "data/processed/comparacao_modelos_v3.csv"
if p.exists():
    registrar("6. COMPARAÇÃO DOS MODELOS",
              pd.read_csv(p).to_string(index=False))

p = ROOT / "data/processed/modelo_ccemg/comparacao_mean_group_ccemg.csv"
if p.exists():
    registrar("7. RESULTADOS MG E CCEMG",
              pd.read_csv(p).to_string(index=False))

p = ROOT / "data/processed/modelo_ccemg/bootstrap_coeficientes_por_pais.csv"
if p.exists():
    registrar("8. BOOTSTRAP DOS COEFICIENTES",
              pd.read_csv(p).to_string(index=False))

# 7. Dashboard
p = ROOT / "src/app.py"
if p.exists():
    codigo = p.read_text(encoding="utf-8", errors="replace")
    termos = [
        "Análise Global", "Evolução Histórica", "Comparar Países",
        "Previsão LSTM", "previsao_lstm.png",
        "ranking_vulnerabilidade_2020.csv"
    ]
    texto = "\n".join(
        f"{'ENCONTRADO' if termo.lower() in codigo.lower() else 'NÃO ENCONTRADO'} | {termo}"
        for termo in termos
    )
    registrar("9. VERIFICAÇÃO DO DASHBOARD", texto)

# 8. Git
# 8. Controle de vers?o
registrar("10. CONTROLE DE VERS?O",
          "Publica??o pelo GitHub Desktop.")

saida = ROOT / "relatorio_auditoria_final.txt"
saida.write_text("\n".join(linhas), encoding="utf-8")

print("\n".join(linhas))
print("\n" + "=" * 75)
print(f"AUDITORIA CONCLUÍDA. RELATÓRIO SALVO EM:\n{saida}")
