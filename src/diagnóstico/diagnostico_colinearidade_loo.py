
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================
# 1. CAMINHOS E DADOS
# ============================================================

RAIZ = Path(__file__).resolve().parents[1]

ENTRADA = (
    RAIZ / "data" / "processed" /
    "modelo_efeitos_ano_fatores_comuns" /
    "amostra_com_medias_loo.csv"
)

SAIDA = (
    RAIZ / "data" / "processed" /
    "diagnostico_colinearidade_loo"
)

SAIDA.mkdir(parents=True, exist_ok=True)

dados = pd.read_csv(ENTRADA)

print("DIAGNÓSTICO DA COLINEARIDADE DAS MÉDIAS LOO")
print(f"Observações: {len(dados)}")
print(f"Países: {dados['Entity'].nunique()}")


# ============================================================
# 2. QUANTIDADE DE PAÍSES POR ANO
# ============================================================

cobertura = (
    dados.groupby("Year")["Entity"]
    .nunique()
    .reset_index(name="numero_paises")
)

print("\nPAÍSES OBSERVADOS POR ANO")
print(cobertura.to_string(index=False))


# ============================================================
# 3. RELAÇÃO ENTRE CADA VARIÁVEL E SUA MÉDIA LOO
# ============================================================

pares = [
    ("delta_mortalidade", "delta_mortalidade_loo"),
    ("delta_combustiveis", "delta_combustiveis_loo"),
    ("delta_log_pib", "delta_log_pib_loo"),
]

resultados = []

for original, loo in pares:

    # Remover efeitos fixos de ano de ambas as variáveis
    x = dados[original] - dados.groupby("Year")[original].transform("mean")
    z = dados[loo] - dados.groupby("Year")[loo].transform("mean")

    correlacao = x.corr(z)

    modelo = sm.OLS(z, sm.add_constant(x)).fit()

    resultados.append({
        "variavel": original,
        "correlacao_apos_efeitos_ano": correlacao,
        "R2_relacao": modelo.rsquared,
        "coeficiente": modelo.params.iloc[1],
        "erro_maximo_relacao_linear": (
            modelo.resid.abs().max()
        ),
    })

diagnostico = pd.DataFrame(resultados)

print("\nRELAÇÃO APÓS RETIRAR EFEITOS DE ANO")
print(diagnostico.to_string(index=False))


# ============================================================
# 4. VERIFICAR VARIAÇÃO DAS MÉDIAS LOO DENTRO DOS ANOS
# ============================================================

variaveis_loo = [
    "delta_mortalidade_loo",
    "delta_combustiveis_loo",
    "delta_log_pib_loo",
]

variacao = []

for coluna in variaveis_loo:

    residual = (
        dados[coluna]
        - dados.groupby("Year")[coluna].transform("mean")
    )

    variacao.append({
        "variavel": coluna,
        "desvio_padrao_original": dados[coluna].std(),
        "desvio_padrao_apos_efeitos_ano": residual.std(),
        "proporcao_variacao_restante": (
            residual.std() / dados[coluna].std()
        ),
    })

variacao = pd.DataFrame(variacao)

print("\nVARIAÇÃO DAS MÉDIAS LOO APÓS EFEITOS DE ANO")
print(variacao.to_string(index=False))


# ============================================================
# 5. EXPORTAÇÃO
# ============================================================

cobertura.to_csv(
    SAIDA / "paises_por_ano.csv",
    index=False
)

diagnostico.to_csv(
    SAIDA / "relacao_variaveis_loo.csv",
    index=False
)

variacao.to_csv(
    SAIDA / "variacao_loo.csv",
    index=False
)

print("\nARQUIVOS SALVOS EM:")
print(SAIDA)

print("\nDIAGNÓSTICO CONCLUÍDO.")
