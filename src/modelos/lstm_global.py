from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------
# CAMINHOS E CONFIGURAÇÕES
# ---------------------------------------------------------

BASE = Path(__file__).resolve().parents[2]
ARQUIVO = BASE / "data" / "processed" / "painel_owid_paises_v2.csv"

SAIDA = BASE / "data" / "processed"
MODELOS = BASE / "models"

SAIDA.mkdir(parents=True, exist_ok=True)
MODELOS.mkdir(parents=True, exist_ok=True)

JANELA = 5
ANO_INICIAL = 1990
ANO_TREINO_FINAL = 2014
ANO_VALIDACAO_FINAL = 2019
ANO_TESTE_FINAL = 2024
EPOCAS = 100

# ---------------------------------------------------------
# CARREGAMENTO E PREPARAÇÃO
# ---------------------------------------------------------

print("\nCARREGANDO BASE...")

df = pd.read_csv(ARQUIVO)

df = df[
    (df["tipo_entidade"] == "codigo_iso3") &
    (df["Year"].between(ANO_INICIAL, ANO_TESTE_FINAL))
].copy()

df["mortalidade_infantil"] = pd.to_numeric(
    df["mortalidade_infantil"], errors="coerce"
)

df = df.dropna(subset=["mortalidade_infantil"])
df = df.sort_values(["Entity", "Year"])

if df.duplicated(["Entity", "Year"]).any():
    raise ValueError("Existem duplicações de país/ano.")

if (df["mortalidade_infantil"] <= 0).any():
    raise ValueError("Foram encontrados valores não positivos.")

# Transformação logarítmica para reduzir assimetria
df["valor"] = np.log1p(df["mortalidade_infantil"])

print("Países com dados:", df["Entity"].nunique())
print("Observações:", len(df))

# ---------------------------------------------------------
# CONSTRUÇÃO DAS SEQUÊNCIAS
# ---------------------------------------------------------

def construir_sequencias(dados, scaler):
    X, y, metadados = [], [], []

    for pais, grupo in dados.groupby("Entity"):
        grupo = grupo.sort_values("Year")

        anos = grupo["Year"].to_numpy()
        valores = grupo["valor"].to_numpy()

        if len(anos) <= JANELA:
            continue

        escalados = scaler.transform(
            valores.reshape(-1, 1)
        ).flatten()

        for i in range(JANELA, len(anos)):
            # Exigir anos consecutivos
            anos_janela = anos[i-JANELA:i+1]

            if not np.all(np.diff(anos_janela) == 1):
                continue

            X.append(escalados[i-JANELA:i])
            y.append(escalados[i])

            metadados.append({
                "pais": pais,
                "ano": int(anos[i]),
                "valor_real": float(
                    np.expm1(valores[i])
                )
            })

    return (
        np.array(X, dtype=np.float32).reshape(-1, JANELA, 1),
        np.array(y, dtype=np.float32),
        pd.DataFrame(metadados)
    )


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


# ---------------------------------------------------------
# SEPARAÇÃO TEMPORAL
# ---------------------------------------------------------

treino = df[df["Year"] <= ANO_TREINO_FINAL].copy()

scaler_teste = MinMaxScaler()
scaler_teste.fit(treino[["valor"]])

X_todos, y_todos, meta_todos = construir_sequencias(
    df, scaler_teste
)

mascara_treino = meta_todos["ano"] <= ANO_TREINO_FINAL
mascara_validacao = (
    (meta_todos["ano"] > ANO_TREINO_FINAL) &
    (meta_todos["ano"] <= ANO_VALIDACAO_FINAL)
)
mascara_teste = (
    (meta_todos["ano"] > ANO_VALIDACAO_FINAL) &
    (meta_todos["ano"] <= ANO_TESTE_FINAL)
)

X_train = X_todos[mascara_treino.to_numpy()]
y_train = y_todos[mascara_treino.to_numpy()]

X_val = X_todos[mascara_validacao.to_numpy()]
y_val = y_todos[mascara_validacao.to_numpy()]

X_test = X_todos[mascara_teste.to_numpy()]
meta_test = meta_todos.loc[mascara_teste].reset_index(drop=True)

print("\nSEQUÊNCIAS:")
print("Treinamento:", len(X_train))
print("Validação:", len(X_val))
print("Teste:", len(X_test))

if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise ValueError("Uma das partições ficou sem sequências.")

# ---------------------------------------------------------
# TREINAMENTO E AVALIAÇÃO
# ---------------------------------------------------------

modelo_teste = criar_modelo()

parada = EarlyStopping(
    monitor="val_loss",
    patience=12,
    restore_best_weights=True
)

print("\nTREINANDO MODELO DE AVALIAÇÃO...")

historico = modelo_teste.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCAS,
    batch_size=32,
    callbacks=[parada],
    verbose=1,
    shuffle=True
)

melhor_epoca = int(
    np.argmin(historico.history["val_loss"]) + 1
)

pred_escalada = modelo_teste.predict(
    X_test, verbose=0
).flatten()

# Converter previsões para a escala original
pred_log = scaler_teste.inverse_transform(
    pred_escalada.reshape(-1, 1)
).flatten()

previsoes = np.maximum(np.expm1(pred_log), 0)

meta_test["previsao_lstm"] = previsoes

# Baseline: repetir o último valor da janela
baseline = []

for _, linha in meta_test.iterrows():
    pais = linha["pais"]
    ano = linha["ano"]

    anterior = df[
        (df["Entity"] == pais) &
        (df["Year"] == ano - 1)
    ]

    baseline.append(
        float(anterior["mortalidade_infantil"].iloc[0])
        if not anterior.empty else np.nan
    )

meta_test["previsao_baseline"] = baseline
meta_test = meta_test.dropna(
    subset=["previsao_baseline"]
)

mae_lstm = mean_absolute_error(
    meta_test["valor_real"],
    meta_test["previsao_lstm"]
)

rmse_lstm = np.sqrt(mean_squared_error(
    meta_test["valor_real"],
    meta_test["previsao_lstm"]
))

mae_baseline = mean_absolute_error(
    meta_test["valor_real"],
    meta_test["previsao_baseline"]
)

rmse_baseline = np.sqrt(mean_squared_error(
    meta_test["valor_real"],
    meta_test["previsao_baseline"]
))

print("\nRESULTADOS FORA DA AMOSTRA")
print("MAE LSTM:", round(mae_lstm, 4))
print("RMSE LSTM:", round(rmse_lstm, 4))
print("MAE baseline:", round(mae_baseline, 4))
print("RMSE baseline:", round(rmse_baseline, 4))

meta_test.to_csv(
    SAIDA / "lstm_avaliacao_2020_2024.csv",
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# TREINAMENTO FINAL COM DADOS ATÉ 2024
# ---------------------------------------------------------

print("\nTREINANDO MODELO FINAL...")

scaler_final = MinMaxScaler()
scaler_final.fit(df[["valor"]])

X_final, y_final, _ = construir_sequencias(
    df, scaler_final
)

modelo_final = criar_modelo()

modelo_final.fit(
    X_final,
    y_final,
    epochs=melhor_epoca,
    batch_size=32,
    verbose=1,
    shuffle=True
)

modelo_final.save(
    MODELOS / "lstm_global.keras"
)

# ---------------------------------------------------------
# PROJEÇÕES RECURSIVAS 2025-2029
# ---------------------------------------------------------

resultados = []

for pais, grupo in df.groupby("Entity"):
    grupo = grupo.sort_values("Year")

    anos = grupo["Year"].to_numpy()
    valores = grupo["valor"].to_numpy()

    if len(anos) < JANELA:
        continue

    # Exigir série terminando em 2024 e janela contínua
    if anos[-1] != ANO_TESTE_FINAL:
        continue

    if not np.all(np.diff(anos[-JANELA:]) == 1):
        continue

    janela = scaler_final.transform(
        valores[-JANELA:].reshape(-1, 1)
    ).flatten().tolist()

    for ano in range(2025, 2030):
        entrada = np.array(
            janela[-JANELA:],
            dtype=np.float32
        ).reshape(1, JANELA, 1)

        pred = modelo_final.predict(
            entrada, verbose=0
        )[0, 0]

        pred_log = scaler_final.inverse_transform(
            np.array([[pred]])
        )[0, 0]

        valor = max(float(np.expm1(pred_log)), 0)

        resultados.append({
            "pais": pais,
            "ano": ano,
            "mortalidade_prevista": valor
        })

        # A previsão alimenta o próximo passo
        janela.append(float(pred))

previsoes_futuras = pd.DataFrame(resultados)

previsoes_futuras.to_csv(
    SAIDA / "lstm_previsoes_2025_2029.csv",
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# SALVAR MÉTRICAS
# ---------------------------------------------------------

metricas = {
    "modelo": "LSTM global univariada",
    "variavel": "Mortalidade de menores de cinco anos",
    "janela_temporal": JANELA,
    "treinamento_ate": ANO_TREINO_FINAL,
    "validacao": "2015-2019",
    "teste": "2020-2024",
    "melhor_epoca": melhor_epoca,
    "mae_lstm": float(mae_lstm),
    "rmse_lstm": float(rmse_lstm),
    "mae_baseline": float(mae_baseline),
    "rmse_baseline": float(rmse_baseline),
    "paises_avaliados": int(meta_test["pais"].nunique()),
    "paises_projetados": int(
        previsoes_futuras["pais"].nunique()
    ),
    "periodo_projecao": "2025-2029"
}

with open(
    SAIDA / "lstm_metricas.json",
    "w",
    encoding="utf-8"
) as arquivo:
    json.dump(
        metricas,
        arquivo,
        ensure_ascii=False,
        indent=4
    )

print("\nPROCESSO CONCLUÍDO")
print("Modelo:", MODELOS / "lstm_global.keras")
print("Avaliação:", SAIDA / "lstm_avaliacao_2020_2024.csv")
print("Projeções:", SAIDA / "lstm_previsoes_2025_2029.csv")
print("Métricas:", SAIDA / "lstm_metricas.json")
print("Países projetados:", previsoes_futuras["pais"].nunique())
