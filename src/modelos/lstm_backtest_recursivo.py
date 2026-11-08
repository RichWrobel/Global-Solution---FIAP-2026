from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers

# =====================================================
# CONFIGURAÇÕES
# =====================================================

BASE = Path(__file__).resolve().parents[2]
ARQUIVO = BASE / "data" / "processed" / "painel_owid_paises_v2.csv"
SAIDA = BASE / "data" / "processed"

JANELA = 5
ANO_INICIAL = 1990
ANO_CORTE = 2019
ANO_FINAL = 2024
EPOCAS = 72
SEMENTE = 42

np.random.seed(SEMENTE)
tf.random.set_seed(SEMENTE)

# =====================================================
# CARREGAMENTO
# =====================================================

print("\nCARREGANDO DADOS...")

df = pd.read_csv(ARQUIVO)

df = df[
    (df["tipo_entidade"] == "codigo_iso3") &
    (df["Year"].between(ANO_INICIAL, ANO_FINAL))
].copy()

df["mortalidade_infantil"] = pd.to_numeric(
    df["mortalidade_infantil"], errors="coerce"
)

df = df.dropna(subset=["mortalidade_infantil"])
df = df.sort_values(["Entity", "Year"])

if df.duplicated(["Entity", "Year"]).any():
    raise ValueError("Existem duplicações de país/ano.")

if (df["mortalidade_infantil"] <= 0).any():
    raise ValueError("Existem valores não positivos.")

df["valor"] = np.log1p(df["mortalidade_infantil"])

# =====================================================
# ESCALONAMENTO SEM VAZAMENTO TEMPORAL
# =====================================================

treino = df[df["Year"] <= ANO_CORTE].copy()

scaler = MinMaxScaler()
scaler.fit(treino[["valor"]])

# =====================================================
# CONSTRUÇÃO DO MODELO
# =====================================================

def criar_modelo():
    modelo = keras.Sequential([
        keras.Input(shape=(JANELA, 1)),
        layers.LSTM(32),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"]
    )

    return modelo

# =====================================================
# SEQUÊNCIAS DE TREINAMENTO
# =====================================================

X, y = [], []

for pais, grupo in treino.groupby("Entity"):
    grupo = grupo.sort_values("Year")

    anos = grupo["Year"].to_numpy()
    valores = grupo["valor"].to_numpy()

    escalados = scaler.transform(
        valores.reshape(-1, 1)
    ).flatten()

    for i in range(JANELA, len(anos)):
        anos_janela = anos[i-JANELA:i+1]

        if not np.all(np.diff(anos_janela) == 1):
            continue

        X.append(escalados[i-JANELA:i])
        y.append(escalados[i])

X = np.array(X, dtype=np.float32).reshape(-1, JANELA, 1)
y = np.array(y, dtype=np.float32)

print("Sequências de treinamento:", len(X))
print("Países disponíveis:", treino["Entity"].nunique())

if len(X) == 0:
    raise ValueError("Não foram construídas sequências de treinamento.")

# =====================================================
# TREINAMENTO ATÉ 2019
# =====================================================

print("\nTREINANDO LSTM ATÉ 2019...")

modelo = criar_modelo()

modelo.fit(
    X,
    y,
    epochs=EPOCAS,
    batch_size=32,
    verbose=1,
    shuffle=True
)

# =====================================================
# BACKTEST RECURSIVO 2020-2024
# =====================================================

print("\nINICIANDO BACKTEST RECURSIVO...")

resultados = []
paises_validos = 0

for pais, grupo in df.groupby("Entity"):
    grupo = grupo.sort_values("Year")

    historico = grupo[grupo["Year"] <= ANO_CORTE]
    futuro_real = grupo[
        grupo["Year"].between(ANO_CORTE + 1, ANO_FINAL)
    ]

    if len(historico) < JANELA:
        continue

    if len(futuro_real) != 5:
        continue

    anos_historico = historico["Year"].to_numpy()

    if not np.all(np.diff(anos_historico[-JANELA:]) == 1):
        continue

    anos_futuros = futuro_real["Year"].to_numpy()

    if not np.array_equal(anos_futuros, np.arange(2020, 2025)):
        continue

    valores_historicos = historico["valor"].to_numpy()

    janela = scaler.transform(
        valores_historicos[-JANELA:].reshape(-1, 1)
    ).flatten().tolist()

    ultimo_valor_real = float(
        historico["mortalidade_infantil"].iloc[-1]
    )

    for ano in range(2020, 2025):

        entrada = np.array(
            janela[-JANELA:],
            dtype=np.float32
        ).reshape(1, JANELA, 1)

        pred_escalada = float(
            modelo.predict(entrada, verbose=0)[0, 0]
        )

        pred_log = float(
            scaler.inverse_transform(
                np.array([[pred_escalada]])
            )[0, 0]
        )

        previsao = max(float(np.expm1(pred_log)), 0)

        valor_real = float(
            futuro_real.loc[
                futuro_real["Year"] == ano,
                "mortalidade_infantil"
            ].iloc[0]
        )

        resultados.append({
            "pais": pais,
            "ano": ano,
            "horizonte": ano - ANO_CORTE,
            "valor_real": valor_real,
            "previsao_lstm": previsao,
            "previsao_baseline": ultimo_valor_real
        })

        # A previsão alimenta o próximo ano.
        # O valor real não entra na janela.
        janela.append(pred_escalada)

    paises_validos += 1

resultado = pd.DataFrame(resultados)

if resultado.empty:
    raise ValueError("Nenhum país pôde ser avaliado.")

# =====================================================
# MÉTRICAS
# =====================================================

def calcular_metricas(dados):
    return {
        "registros": int(len(dados)),
        "paises": int(dados["pais"].nunique()),
        "mae_lstm": float(mean_absolute_error(
            dados["valor_real"], dados["previsao_lstm"]
        )),
        "rmse_lstm": float(np.sqrt(mean_squared_error(
            dados["valor_real"], dados["previsao_lstm"]
        ))),
        "mae_baseline": float(mean_absolute_error(
            dados["valor_real"], dados["previsao_baseline"]
        )),
        "rmse_baseline": float(np.sqrt(mean_squared_error(
            dados["valor_real"], dados["previsao_baseline"]
        )))
    }

metricas_gerais = calcular_metricas(resultado)

metricas_anuais = []

for ano, grupo in resultado.groupby("ano"):
    linha = {"ano": int(ano)}
    linha.update(calcular_metricas(grupo))
    metricas_anuais.append(linha)

metricas_anuais = pd.DataFrame(metricas_anuais)

# =====================================================
# SALVAMENTO
# =====================================================

resultado.to_csv(
    SAIDA / "lstm_backtest_recursivo_2020_2024.csv",
    index=False,
    encoding="utf-8-sig"
)

metricas_anuais.to_csv(
    SAIDA / "lstm_backtest_metricas_anuais.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    SAIDA / "lstm_backtest_metricas.json",
    "w",
    encoding="utf-8"
) as arquivo:
    json.dump(
        metricas_gerais,
        arquivo,
        ensure_ascii=False,
        indent=4
    )

# =====================================================
# RESULTADOS
# =====================================================

print("\n======================================")
print("BACKTEST RECURSIVO — RESULTADO GERAL")
print("======================================")

for chave, valor in metricas_gerais.items():
    print(f"{chave}: {valor}")

print("\n======================================")
print("RESULTADOS POR ANO")
print("======================================")

print(metricas_anuais.round(4).to_string(index=False))

print("\nPaíses avaliados:", paises_validos)
print("Registros avaliados:", len(resultado))
print("\nARQUIVOS SALVOS:")
print("lstm_backtest_recursivo_2020_2024.csv")
print("lstm_backtest_metricas_anuais.csv")
print("lstm_backtest_metricas.json")
print("\nBACKTEST CONCLUÍDO.")
