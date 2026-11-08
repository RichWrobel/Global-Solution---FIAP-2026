import pandas as pd

arquivo = "data/processed/modelo_ccemg/coeficientes_por_pais.csv"
df = pd.read_csv(arquivo)

cenarios = {
    "Amostra completa": [],
    "Sem Haiti": ["Haiti"],
    "Sem Somália": ["Somalia"],
    "Sem Haiti e Somália": ["Haiti", "Somalia"]
}

resultados = []

for (modelo, limite), grupo in df.groupby(
    ["modelo", "limite_minimo_observacoes"]
):
    for nome_cenario, excluir in cenarios.items():
        amostra = grupo[~grupo["pais"].isin(excluir)]
        coef = amostra["coef_combustiveis"].dropna()

        resultados.append({
            "modelo": modelo,
            "limite": limite,
            "cenario": nome_cenario,
            "paises": len(coef),
            "media": coef.mean(),
            "mediana": coef.median(),
            "desvio_padrao": coef.std(),
            "percentual_positivo": (coef > 0).mean() * 100,
            "percentual_negativo": (coef < 0).mean() * 100
        })

resultado = pd.DataFrame(resultados)

print("\nAUDITORIA DE INFLUÊNCIA — COMBUSTÍVEIS LIMPOS\n")
print(resultado.round(4).to_string(index=False))

saida = "data/processed/modelo_ccemg/auditoria_influencia_haiti_somalia.csv"
resultado.to_csv(saida, index=False, encoding="utf-8-sig")

print(f"\nArquivo salvo em: {saida}")
