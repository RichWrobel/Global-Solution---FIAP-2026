import os
import time
import pandas as pd
import requests

print("Iniciando o download dos dados do OWID...")

SLUGS = {
    "combustiveis_limpos": "access-to-clean-fuels-and-technologies-for-cooking",
    "mortalidade_infantil": "child-mortality",
    "pib_per_capita": "gdp-per-capita-worldbank-constant-usd",
    "acesso_eletricidade": "share-of-the-population-with-access-to-electricity",
    "escolaridade": "mean-years-of-schooling-long-run",
}

BASE_URL = "https://ourworldindata.org/grapher"
PASTA_RAW = "data/raw"
PASTA_META = "data/raw/metadata"

# Define o cabeçalho para ser identificado como um cliente legítimo
HEADERS = {
    "User-Agent": "Our World In Data data fetch/1.0"
}

os.makedirs(PASTA_RAW, exist_ok=True)
os.makedirs(PASTA_META, exist_ok=True)

for nome, slug in SLUGS.items():
    print(f"Baixando {nome}...")
    url_csv = f"{BASE_URL}/{slug}.csv?v=1&csvType=full&useColumnShortNames=false"
    try:
        # AQUI ESTÁ A CORREÇÃO: passamos o HEADERS para o pandas
        df = pd.read_csv(url_csv, storage_options=HEADERS)
        caminho_csv = f"{PASTA_RAW}/{nome}.csv"
        df.to_csv(caminho_csv, index=False)
        print(f"[OK] CSV: {nome} ({df.shape[0]} linhas, {df.shape[1]} colunas)")
    except Exception as e:
        print(f"[ERRO] CSV {nome}: {e}")
        continue

    url_meta = f"{BASE_URL}/{slug}.metadata.json?v=1&csvType=full&useColumnShortNames=false"
    try:
        # Para o requests, também aplicamos o cabeçalho
        r = requests.get(url_meta, headers=HEADERS, timeout=30)
        r.raise_for_status()
        caminho_meta = f"{PASTA_META}/{nome}.metadata.json"
        with open(caminho_meta, "w", encoding="utf-8") as f:
            f.write(r.text)
        print(f"[OK] Metadados: {nome}")
    except Exception as e:
        print(f"[AVISO] Metadados {nome}: {e}")

    time.sleep(1)

print("\nDownload finalizado com sucesso!")