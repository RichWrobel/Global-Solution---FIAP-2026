
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

# ============================================================
# 1. CAMINHOS
# ============================================================

BASE = Path(__file__).resolve().parents[1]

ARQUIVO = (
    BASE / "data" / "processed"
    / "diagnostico_residuos_ccemg"
    / "residuos_ccemg.csv"
)

PASTA_SAIDA = (
    BASE / "data" / "processed"
    / "teste_cd_residuos_ccemg"
)

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. CARREGAMENTO E ORGANIZAÇÃO
# ============================================================

df = pd.read_csv(ARQUIVO)

df = df[["pais", "Year", "residuo"]].dropna()

painel = df.pivot(
    index="Year",
    columns="pais",
    values="residuo"
)

paises = painel.columns.tolist()
n = len(paises)

# ============================================================
# 3. CORRELAÇÕES PAREADAS
# ============================================================

correlacoes = painel.corr(min_periods=10)

presenca = painel.notna().astype(int)
contagens = presenca.T @ presenca

pares = []

for i in range(n):
    for j in range(i + 1, n):

        rho = correlacoes.iloc[i, j]
        t_ij = contagens.iloc[i, j]

        if pd.notna(rho) and t_ij >= 10:
            pares.append({
                "pais_1": paises[i],
                "pais_2": paises[j],
                "correlacao": rho,
                "observacoes_comuns": int(t_ij)
            })

pares_df = pd.DataFrame(pares)

if pares_df.empty:
    raise ValueError("Não foram encontrados pares válidos.")

# ============================================================
# 4. ESTATÍSTICA CD - APROXIMAÇÃO DIAGNÓSTICA
# ============================================================

soma = (
    np.sqrt(pares_df["observacoes_comuns"])
    * pares_df["correlacao"]
).sum()

estatistica_cd = np.sqrt(
    2 / (n * (n - 1))
) * soma

p_valor = 2 * norm.sf(abs(estatistica_cd))

# ============================================================
# 5. RESUMO
# ============================================================

resultado = pd.DataFrame([{
    "paises": n,
    "pares_validos": len(pares_df),
    "observacoes_minimas_por_par": 10,
    "estatistica_CD": estatistica_cd,
    "p_valor": p_valor,
    "correlacao_media": pares_df["correlacao"].mean(),
    "correlacao_mediana": pares_df["correlacao"].median(),
    "correlacao_media_absoluta": pares_df["correlacao"].abs().mean(),
    "percentual_positivas": (
        pares_df["correlacao"].gt(0).mean() * 100
    )
}])

resultado.to_csv(
    PASTA_SAIDA / "resultado_teste_cd.csv",
    index=False
)

pares_df.to_csv(
    PASTA_SAIDA / "correlacoes_pareadas.csv",
    index=False
)

# ============================================================
# 6. RESULTADOS
# ============================================================

print("=" * 75)
print("TESTE CD DE PESARAN - RESÍDUOS CCEMG")
print("=" * 75)

print(resultado.to_string(index=False))

print("\nINTERPRETAÇÃO CONVENCIONAL")

if p_valor < 0.01:
    print("Rejeita-se H0 a 1%: evidência de dependência transversal.")
elif p_valor < 0.05:
    print("Rejeita-se H0 a 5%: evidência de dependência transversal.")
elif p_valor < 0.10:
    print("Rejeita-se H0 a 10%: evidência de dependência transversal.")
else:
    print("Não se rejeita H0 nos níveis convencionais.")

print("\nARQUIVOS GERADOS:")
print(PASTA_SAIDA / "resultado_teste_cd.csv")
print(PASTA_SAIDA / "correlacoes_pareadas.csv")

print("\nNOTA METODOLÓGICA:")
print(
    "A fórmula convencional do CD requer cautela neste painel "
    "desbalanceado, com diferentes períodos de sobreposição. "
    "O resultado é preliminar e não constitui validação definitiva."
)

print("\nDIAGNÓSTICO CONCLUÍDO.")
