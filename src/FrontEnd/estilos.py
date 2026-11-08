
import streamlit as st
import plotly.io as pio


# ==========================================
# IDENTIDADE VISUAL
# O PREÇO OCULTO DA ENERGIA SUJA
# ==========================================

CORES = {
    "fundo": "#0B1220",
    "superficie": "#111C2E",
    "cartao": "#162438",
    "borda": "#263B55",
    "texto": "#F1F5F9",
    "texto_secundario": "#94A3B8",
    "verde": "#2DD4BF",
    "verde_escuro": "#0F766E",
    "azul": "#38BDF8",
    "laranja": "#F59E0B",
    "vermelho": "#F87171",
}


def aplicar_estilos():

    # Tema global dos gráficos Plotly
    pio.templates.default = "plotly_dark"

    # Aplicação do CSS
    st.markdown(
        """
        <style>

        /* FUNDO GERAL */
        .stApp {
            background-color: #0B1220;
            color: #F1F5F9;
        }

        /* ÁREA PRINCIPAL */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        /* TIPOGRAFIA */
        h1, h2, h3 {
            color: #F8FAFC !important;
            font-weight: 700 !important;
        }

        p, label, li {
            color: #CBD5E1;
        }

        /* BARRA LATERAL */
        [data-testid="stSidebar"] {
            background-color: #111C2E;
            border-right: 1px solid #263B55;
        }

        /* CARTÕES DE MÉTRICAS */
        [data-testid="stMetric"] {
            background-color: #162438;
            border: 1px solid #263B55;
            padding: 20px;
            border-radius: 14px;
        }

        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
        }

        [data-testid="stMetricValue"] {
            color: #2DD4BF !important;
            font-weight: 700;
        }

        /* ABAS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 1px solid #263B55;
        }

        .stTabs [data-baseweb="tab"] {
            color: #94A3B8;
            font-weight: 600;
            padding: 12px 18px;
        }

        .stTabs [aria-selected="true"] {
            color: #2DD4BF !important;
        }

        /* SELETORES */
        div[data-baseweb="select"] > div {
            background-color: #162438;
            border-color: #263B55;
            border-radius: 8px;
        }

        /* TABELAS */
        [data-testid="stDataFrame"] {
            border: 1px solid #263B55;
            border-radius: 12px;
            overflow: hidden;
        }

        /* DIVISORES */
        hr {
            border-color: #263B55;
        }

        /* MENSAGENS */
        [data-testid="stAlert"] {
            border-radius: 10px;
        }

        /* RODAPÉ */
        footer {
            visibility: hidden;
        }

        </style>
        """,
        unsafe_allow_html=True
    )
