import pandas as pd
import numpy as np

arquivo = "data/processed/modelo_ccemg/coeficientes_por_pais.csv"
df = pd.read_csv(arquivo)

N_BOOT = 10000
SEMENTE = 42

rng = np.random.default_rng(SEMENTE)
resultados = []

for (modelo, limite), grupo in df.groupby(
    ["modelo", "limite_minimo_observacoes"]
):
    valores = grupo["coef_combustiveis"].dropna().to_numpy()
    n = len(valores)

    medias_bootstrap = np.empty(N_BOOT)

    for i in range(N_BOOT):
        amostra = rng.choice(valores, size=n, replace=True)
        medias_bootstrap[i] = np.mean(amostra)

    media_original = np.mean(valores)
    limite_inferior = np.percentile(medias_bootstrap, 2.5)
    limite_superior = np.percentile(medias_bootstrap, 97.5)

    resultados.append({
        "modelo": modelo,
        "limite_minimo": limite,
        "paises": n,
        "media_original": media_original,
        "ic95_inferior": limite_inferior,
        "ic95_superior": limite_superior,
        "inclui_zero": limite_inferior <= 0 <= limite_superior
    })

resultado = pd.DataFrame(resultados)

print("\nBOOTSTRAP DOS COEFICIENTES POR PAÍS")
print("10.000 reamostragens | Intervalo percentil de 95%\n")
print(resultado.round(6).to_string(index=False))

saida = (
    "data/processed/modelo_ccemg/"
    "bootstrap_coeficientes_por_pais.csv"
)

resultado.to_csv(saida, index=False, encoding="utf-8-sig")

print(f"\nArquivo salvo em: {saida}")
