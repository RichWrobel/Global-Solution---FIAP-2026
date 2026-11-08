import streamlit as st


def exibir_kpi(titulo, valor, descricao=None):
    """
    Exibe um indicador visual (KPI) no dashboard.
    """
    with st.container(border=True):
        st.metric(
            label=titulo,
            value=valor,
            help=descricao
        )