
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except ImportError:
    print("ERRO: statsmodels não está instalado.")
    print("Execute: python -m pip install statsmodels")
    sys.exit(1)


# ============================================================
# 1. CAMINHOS E CONFIGURAÇÕES
# ============================================================

BASE = Path(__file__).resolve().parents[1]

ARQUIVO_ENTRADA = (
    BASE / "data" / "processed" / "painel_owid_integrado.csv"
)

PASTA_SAIDA = BASE / "data" / "processed" / "modelos"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

ARQUIVO_RESULTADOS = PASTA_SAIDA / "modelo_efeitos_fixos.txt"
ARQUIVO_COEFICIENTES = PASTA_SAIDA / "coeficientes_efeitos_fixos.csv"
ARQUIVO_DIAGNOSTICO = PASTA_SAIDA / "diagnostico_modelo.csv"


# ============================================================
# 2. CARREGAMENTO E PREPARAÇÃO
# ============================================================

print("=" * 70)
print("MODELO DE REGRESSÃO COM EFEITOS FIXOS")
print("=" * 70)

if not ARQUIVO_ENTRADA.exists():
    print(f"Arquivo não encontrado: {ARQUIVO_ENTRADA}")
    sys.exit(1)

df = pd.read_csv(ARQUIVO_ENTRADA)

colunas_necessarias = [
    "Entity",
    "Year",
    "combustiveis_limpos",
    "mortalidade_infantil",
    "pib_per_capita",
]

faltantes = [
    coluna for coluna in colunas_necessarias
    if coluna not in df.columns
]

if faltantes:
    print("Colunas ausentes:", faltantes)
    print("Colunas disponíveis:", list(df.columns))
    sys.exit(1)

df = df[colunas_necessarias].copy()

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

for coluna in [
    "combustiveis_limpos",
    "mortalidade_infantil",
    "pib_per_capita",
]:
    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

df = df.dropna(subset=["Entity", "Year"])
df = df[df["pib_per_capita"] > 0]

# Logaritmo natural do PIB per capita
df["log_pib_per_capita"] = np.log(df["pib_per_capita"])

# Amostra completa para o modelo
amostra = df.dropna(
    subset=[
        "combustiveis_limpos",
        "mortalidade_infantil",
        "log_pib_per_capita",
    ]
).copy()

amostra["Year"] = amostra["Year"].astype(int)

# Verificação de duplicidades por país/ano
duplicidades = amostra.duplicated(
    subset=["Entity", "Year"]
).sum()

print(f"\nRegistros na base original: {len(df)}")
print(f"Observações completas: {len(amostra)}")
print(f"Entidades: {amostra['Entity'].nunique()}")
print(
    f"Período: {amostra['Year'].min()}–"
    f"{amostra['Year'].max()}"
)
print(f"Duplicidades país/ano: {duplicidades}")

if duplicidades > 0:
    print(
        "\nERRO: existem observações duplicadas por entidade e ano."
    )
    print(
        "Precisamos investigar essas duplicidades antes da regressão."
    )
    sys.exit(1)

if amostra["Entity"].nunique() < 2:
    print("ERRO: número insuficiente de entidades.")
    sys.exit(1)


# ============================================================
# 3. ESTIMAÇÃO DO MODELO
# ============================================================

print("\nEstimando modelo...")

formula = (
    "mortalidade_infantil ~ "
    "combustiveis_limpos + "
    "log_pib_per_capita + "
    "C(Entity) + C(Year)"
)

modelo = smf.ols(
    formula=formula,
    data=amostra
).fit(
    cov_type="cluster",
    cov_kwds={"groups": amostra["Entity"]}
)

# ============================================================
# 4. RESULTADOS
# ============================================================

print("\n" + "=" * 70)
print("RESULTADO DA REGRESSÃO")
print("=" * 70)

print(modelo.summary())

with open(ARQUIVO_RESULTADOS, "w", encoding="utf-8") as arquivo:
    arquivo.write(str(modelo.summary()))

# ============================================================
# 5. COEFICIENTES PRINCIPAIS
# ============================================================

variaveis_principais = [
    "combustiveis_limpos",
    "log_pib_per_capita",
]

resultados = []

for variavel in variaveis_principais:
    resultados.append({
        "variavel": variavel,
        "coeficiente": modelo.params.get(variavel, np.nan),
        "erro_padrao": modelo.bse.get(variavel, np.nan),
        "valor_p": modelo.pvalues.get(variavel, np.nan),
        "intervalo_inferior": modelo.conf_int().loc[
            variavel, 0
        ],
        "intervalo_superior": modelo.conf_int().loc[
            variavel, 1
        ],
    })

coeficientes = pd.DataFrame(resultados)

coeficientes.to_csv(
    ARQUIVO_COEFICIENTES,
    index=False,
    encoding="utf-8-sig"
)

print("\nCOEFICIENTES PRINCIPAIS:")
print(coeficientes.to_string(index=False))

# ============================================================
# 6. DIAGNÓSTICO DA AMOSTRA
# ============================================================

diagnostico = pd.DataFrame([{
    "observacoes": len(amostra),
    "entidades": amostra["Entity"].nunique(),
    "anos": amostra["Year"].nunique(),
    "ano_inicial": amostra["Year"].min(),
    "ano_final": amostra["Year"].max(),
    "duplicidades": duplicidades,
    "r2": modelo.rsquared,
    "r2_ajustado": modelo.rsquared_adj,
    "f_estatistica": modelo.fvalue,
    "observacoes_por_entidade_media": (
        len(amostra) / amostra["Entity"].nunique()
    ),
}])

diagnostico.to_csv(
    ARQUIVO_DIAGNOSTICO,
    index=False,
    encoding="utf-8-sig"
)

print("\nARQUIVOS GERADOS:")
print(ARQUIVO_RESULTADOS)
print(ARQUIVO_COEFICIENTES)
print(ARQUIVO_DIAGNOSTICO)

print("\nMODELO CONCLUÍDO.")
