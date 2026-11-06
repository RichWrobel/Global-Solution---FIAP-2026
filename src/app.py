import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Configuração da página
st.set_page_config(page_title="Energia Limpa x Mortalidade", layout="wide")

st.title("🌍 O Preço Oculto da Energia Suja")
st.markdown("""
**Objetivo:** Investigar o impacto do acesso a combustíveis limpos na mortalidade infantil global.
*Dados extraídos do Our World in Data (OWID) | Modelos: Random Forest, K-Means e Redes Neurais LSTM*
""")

# Funções para carregar dados

def load_clusters():
    caminho = "data/processed/painel_clusters.csv"
    if os.path.exists(caminho):
        return pd.read_csv(caminho)
    return pd.DataFrame()

df_clusters = load_clusters()

# Criando as abas
tab1, tab2 = st.tabs(["🌎 Análise Global (Clusters)", "🧠 Previsão LSTM"])

# --- ABA 1: CLUSTERS ---
with tab1:
    st.header("Mapeamento Global de Intervenção")
    st.markdown("A Inteligência Artificial (K-Means) agrupou os países em 3 perfis para orientar políticas públicas.")
    
    if not df_clusters.empty:
        fig_cluster = px.scatter(
            df_clusters, 
            x="combustiveis_limpos", 
            y="mortalidade_infantil",
            color="perfil",
            hover_name="Pais",
            color_discrete_map={
                '1. Alta Prioridade (Baixo Acesso)': '#e74c3c',  # Vermelho
                '2. Em Transição': '#f1c40f',                    # Amarelo
                '3. Consolidado (Alto Acesso)': '#2ecc71'        # Verde
            },
            labels={
                "combustiveis_limpos": "Acesso a Combustíveis Limpos (%)",
                "mortalidade_infantil": "Mortalidade Infantil (por 1.000)",
                "perfil": "Perfil de Intervenção"
            }
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        st.info("💡 **Insight:** O grupo em vermelho indica os países onde a falta de energia limpa é mais letal. Investimentos nestes locais têm maior impacto social.")
    else:
        st.warning("⚠️ Arquivo de clusters não encontrado. Rode a célula de Clusterização (K-Means) no notebook para gerar o arquivo 'painel_clusters.csv' na pasta data/processed.")

# --- ABA 2: LSTM ---
with tab2:
    st.header("Previsão com Redes Neurais (LSTM)")
    st.markdown("A Rede Neural Recorrente analisa o histórico e projeta os próximos anos para apoiar o planejamento do sistema de saúde.")
    
    caminho_imagem = "figuras/previsao_lstm.png"
    
    # Tenta carregar a imagem de previsão
    if os.path.exists(caminho_imagem):
        st.image(caminho_imagem, caption="Previsão da Mortalidade Infantil no Brasil (LSTM)")
        st.info("💡 **Insight:** O modelo aponta a tendência de estabilização/queda. Na nuvem (AWS), esse modelo rodaria diariamente para atualizar as projeções para os gestores.")
    else:
        st.warning("⚠️ Imagem de previsão não encontrada. Rode a célula da LSTM no notebook para gerar o gráfico e salvá-lo na pasta figuras/.")