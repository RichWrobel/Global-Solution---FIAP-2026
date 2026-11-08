
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from FrontEnd.componentes import exibir_kpi
from FrontEnd.estilos import aplicar_estilos
from FrontEnd.grafico import (
    grafico_energia_mortalidade,
    grafico_mapa_mundi
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="O Preço Oculto da Energia Suja",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

aplicar_estilos()

st.title("🌍 O Preço Oculto da Energia Suja")

st.markdown("""
**Energia limpa e saúde infantil**

Uma ferramenta de análise de dados para investigar padrões
entre acesso à energia, condições socioeconômicas e
mortalidade infantil.
""")


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

@st.cache_data
def carregar_clusters():
    caminho = BASE / "data" / "processed" / "painel_clusters.csv"

    if caminho.exists():
        return pd.read_csv(caminho)

    return pd.DataFrame()


@st.cache_data
def carregar_historico():
    caminho = BASE / "data" / "processed" / "painel_owid_paises_v2.csv"

    if caminho.exists():
        return pd.read_csv(caminho)

    return pd.DataFrame()


df_clusters = carregar_clusters()
df_historico = carregar_historico()

if not df_historico.empty:
    st.sidebar.success(
        f"Base histórica carregada: {len(df_historico):,} registros"
    )
else:
    st.sidebar.warning("Base histórica não encontrada.")


# ============================================================
# ABAS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🌎 Análise Global",
    "📈 Evolução Histórica",
    "🔎 Comparar Países",
    "🧠 Previsão LSTM"
])


# ============================================================
# ABA 1 - ANÁLISE GLOBAL
# ============================================================

with tab1:

    st.header("Mapeamento Global de Vulnerabilidade")

    if not df_clusters.empty:

        # ----------------------------------------------------
        # FILTROS
        # ----------------------------------------------------

        st.sidebar.header("Filtros de análise")

        perfis = sorted(
            df_clusters["perfil"].dropna().unique()
        )

        perfis_selecionados = st.sidebar.multiselect(
            "Perfis de intervenção",
            options=perfis,
            default=perfis
        )

        df_filtrado = df_clusters[
            df_clusters["perfil"].isin(perfis_selecionados)
        ].copy()

        # ----------------------------------------------------
        # INDICADORES GERAIS
        # ----------------------------------------------------

        total_paises = df_filtrado["Pais"].nunique()

        alta_prioridade = df_filtrado[
            df_filtrado["perfil"].str.contains(
                "Alta Prioridade",
                case=False,
                na=False
            )
        ]["Pais"].nunique()

        em_transicao = df_filtrado[
            df_filtrado["perfil"].str.contains(
                "Em Transição",
                case=False,
                na=False
            )
        ]["Pais"].nunique()

        consolidado = df_filtrado[
            df_filtrado["perfil"].str.contains(
                "Consolidado",
                case=False,
                na=False
            )
        ]["Pais"].nunique()

        media_combustiveis = df_filtrado[
            "combustiveis_limpos"
        ].mean()

        media_mortalidade = df_filtrado[
            "mortalidade_infantil"
        ].mean()

        # ----------------------------------------------------
        # PERFIS DE INTERVENÇÃO
        # ----------------------------------------------------

        st.subheader("Perfis de intervenção")

        col1, col2, col3 = st.columns(3)

        with col1:
            exibir_kpi("Alta prioridade", alta_prioridade)

        with col2:
            exibir_kpi("Em transição", em_transicao)

        with col3:
            exibir_kpi("Consolidado", consolidado)

        st.divider()

        # ----------------------------------------------------
        # INDICADORES GERAIS
        # ----------------------------------------------------

        st.subheader("Indicadores gerais")

        col1, col2, col3 = st.columns(3)

        with col1:
            exibir_kpi("Países analisados", total_paises)

        with col2:
            valor_combustiveis = (
                f"{media_combustiveis:.2f}%"
                if pd.notna(media_combustiveis)
                else "Sem dados"
            )

            exibir_kpi(
                "Acesso médio a combustíveis limpos",
                valor_combustiveis
            )

        with col3:
            valor_mortalidade = (
                f"{media_mortalidade:.2f}"
                if pd.notna(media_mortalidade)
                else "Sem dados"
            )

            exibir_kpi(
                "Mortalidade infantil média",
                valor_mortalidade
            )

        st.divider()

        # ----------------------------------------------------
        # MAPA-MÚNDI INTERATIVO
        # ----------------------------------------------------

        st.subheader(
            "Distribuição geográfica dos perfis de intervenção"
        )

        st.markdown(
            "Explore a distribuição dos países conforme seu nível "
            "de acesso à energia limpa e suas condições socioeconômicas. "
            "Passe o cursor sobre os países para consultar os indicadores."
        )

        fig_mapa = grafico_mapa_mundi(df_filtrado)

        if fig_mapa is not None:

            st.plotly_chart(
                fig_mapa,
                width="stretch"
            )

            st.caption(
                "Classificação dos países referente a 2020. "
                "Os perfis representam agrupamentos estatísticos "
                "e não estabelecem relações causais."
            )

        else:
            st.warning(
                "Não existem dados suficientes para gerar o mapa."
            )

        st.divider()

        # ----------------------------------------------------
        # PAINEL DE ANÁLISE INDIVIDUAL
        # ----------------------------------------------------

        st.subheader("Análise individual por país")

        st.markdown(
            "Selecione um país para consultar seus indicadores "
            "e compará-los com a média da amostra analisada."
        )

        paises_disponiveis = sorted(
            df_filtrado["Pais"].dropna().unique()
        )

        if paises_disponiveis:

            pais_selecionado = st.selectbox(
                "Selecione um país",
                paises_disponiveis,
                index=(
                    paises_disponiveis.index("Brazil")
                    if "Brazil" in paises_disponiveis
                    else 0
                ),
                key="pais_analise_individual"
            )

            dados_pais = df_filtrado[
                df_filtrado["Pais"] == pais_selecionado
            ].iloc[0]

            st.markdown(f"### {pais_selecionado}")

            st.info(
                f"Perfil de intervenção: {dados_pais['perfil']}"
            )

            indicadores_pais = [
                (
                    "Acesso a combustíveis limpos",
                    "combustiveis_limpos",
                    "%"
                ),
                (
                    "Acesso à eletricidade",
                    "acesso_eletricidade",
                    "%"
                ),
                (
                    "Mortalidade infantil",
                    "mortalidade_infantil",
                    "por 1.000 nascidos vivos"
                ),
                (
                    "PIB per capita",
                    "pib_per_capita",
                    "US$"
                )
            ]

            colunas_kpi = st.columns(4)

            for coluna, (titulo, indicador, unidade) in zip(
                colunas_kpi,
                indicadores_pais
            ):

                valor = dados_pais[indicador]

                if pd.notna(valor):

                    if unidade == "%":
                        valor_formatado = f"{valor:.2f}%"

                    elif unidade == "US$":
                        valor_formatado = f"US$ {valor:,.2f}"

                    else:
                        valor_formatado = f"{valor:.2f}"

                else:
                    valor_formatado = "Sem dados"

                with coluna:
                    exibir_kpi(
                        titulo,
                        valor_formatado
                    )

            st.divider()

            st.markdown(
                "#### Comparação com a média da amostra"
            )

            comparacao = []

            for titulo, indicador, unidade in indicadores_pais:

                media = df_filtrado[indicador].mean()
                valor = dados_pais[indicador]

                if pd.notna(valor) and pd.notna(media):

                    comparacao.append({
                        "Indicador": titulo,
                        "País selecionado": valor,
                        "Média da amostra": media,
                        "Diferença": valor - media
                    })

            if comparacao:

                df_comparacao = pd.DataFrame(comparacao)

                st.dataframe(
                    df_comparacao,
                    column_config={
                        "País selecionado": st.column_config.NumberColumn(
                            "País selecionado",
                            format="%.2f"
                        ),
                        "Média da amostra": st.column_config.NumberColumn(
                            "Média da amostra",
                            format="%.2f"
                        ),
                        "Diferença": st.column_config.NumberColumn(
                            "Diferença",
                            format="%.2f"
                        )
                    },
                    width="stretch",
                    hide_index=True
                )

            else:
                st.warning(
                    "Não existem dados suficientes para comparação."
                )

        else:
            st.warning(
                "Não existem países disponíveis para análise."
            )

        st.divider()

        # ----------------------------------------------------
        # RANKING DE VULNERABILIDADE
        # ----------------------------------------------------

        st.subheader(
            "Onde a intervenção é mais urgente?"
        )

        st.markdown("""
        Ranking exploratório que combina três dimensões:
        acesso a combustíveis limpos, acesso à eletricidade
        e mortalidade infantil.
        """)

        colunas_ranking = [
            "Pais",
            "combustiveis_limpos",
            "acesso_eletricidade",
            "mortalidade_infantil",
            "perfil"
        ]

        # O ranking utiliza a base original, sem depender
        # dos filtros de perfil selecionados na barra lateral.

        ranking = df_clusters[
            colunas_ranking
        ].dropna(
            subset=[
                "Pais",
                "combustiveis_limpos",
                "acesso_eletricidade",
                "mortalidade_infantil"
            ]
        ).copy()

        if not ranking.empty:

            # MENOR ACESSO = MAIOR VULNERABILIDADE

            ranking["score_combustiveis"] = (
                1 - ranking["combustiveis_limpos"].rank(
                    pct=True,
                    ascending=True
                )
            ) * 100

            ranking["score_eletricidade"] = (
                1 - ranking["acesso_eletricidade"].rank(
                    pct=True,
                    ascending=True
                )
            ) * 100

            # MAIOR MORTALIDADE = MAIOR VULNERABILIDADE

            ranking["score_mortalidade"] = (
                ranking["mortalidade_infantil"].rank(
                    pct=True,
                    ascending=True
                )
            ) * 100

            # ÍNDICE COMPOSTO: MÉDIA DOS TRÊS PERCENTIS

            ranking["indice_vulnerabilidade"] = ranking[
                [
                    "score_combustiveis",
                    "score_eletricidade",
                    "score_mortalidade"
                ]
            ].mean(axis=1)

            ranking = ranking.sort_values(
                "indice_vulnerabilidade",
                ascending=False
            ).reset_index(drop=True)

            ranking["Posição"] = ranking.index + 1

            # INDICADORES DO RANKING

            col1, col2, col3 = st.columns(3)

            with col1:
                exibir_kpi(
                    "Países classificados",
                    len(ranking)
                )

            with col2:
                exibir_kpi(
                    "Maior vulnerabilidade",
                    ranking.iloc[0]["Pais"]
                )

            with col3:
                exibir_kpi(
                    "Índice máximo",
                    f"{ranking.iloc[0]['indice_vulnerabilidade']:.2f}/100"
                )

            st.divider()

            # TOP 10 - MAIOR VULNERABILIDADE

            st.markdown(
                "#### Top 10 — Maior vulnerabilidade combinada"
            )

            top10 = ranking.head(10).copy()

            dados_grafico = top10.sort_values(
                "indice_vulnerabilidade",
                ascending=True
            )

            fig_ranking = px.bar(
                dados_grafico,
                x="indice_vulnerabilidade",
                y="Pais",
                orientation="h",
                color="indice_vulnerabilidade",
                color_continuous_scale="Reds",
                text="indice_vulnerabilidade",
                labels={
                    "indice_vulnerabilidade": "Índice de vulnerabilidade",
                    "Pais": "País"
                },
                title="Dez países com maior vulnerabilidade relativa"
            )

            fig_ranking.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            fig_ranking.update_layout(
                xaxis_range=[0, 110],
                coloraxis_showscale=False,
                height=480,
                autosize=True,
                font=dict(size=10),
                title=dict(font=dict(size=16)),
                xaxis=dict(
                    tickfont=dict(size=10),
                    automargin=True
                ),
                yaxis=dict(
                    tickfont=dict(size=10),
                    automargin=True
                ),
                margin=dict(l=20, r=35, t=65, b=35)
            )

            st.plotly_chart(
                fig_ranking,
                width="stretch"
            )

            st.divider()

            # TOP 10 - COMBUSTÍVEIS LIMPOS

            st.markdown(
                "#### Países com menor acesso a combustíveis limpos"
            )

            top_combustiveis = ranking.nsmallest(
                10,
                "combustiveis_limpos"
            )

            tabela_combustiveis = top_combustiveis[
                [
                    "Pais",
                    "combustiveis_limpos",
                    "perfil"
                ]
            ].rename(
                columns={
                    "Pais": "País",
                    "combustiveis_limpos": "Acesso (%)",
                    "perfil": "Perfil de intervenção"
                }
            )

            st.dataframe(
                tabela_combustiveis,
                column_config={
                    "País": st.column_config.TextColumn(
                        "País",
                        width="medium"
                    ),
                    "Acesso (%)": st.column_config.NumberColumn(
                        "Acesso (%)",
                        format="%.2f",
                        width="medium"
                    ),
                    "Perfil de intervenção": st.column_config.TextColumn(
                        "Perfil de intervenção",
                        width="large"
                    )
                },
                width="stretch",
                hide_index=True
            )

            st.divider()

            # TOP 10 - MORTALIDADE INFANTIL

            st.markdown(
                "#### Países com maior mortalidade infantil"
            )

            top_mortalidade = ranking.nlargest(
                10,
                "mortalidade_infantil"
            )

            tabela_mortalidade = top_mortalidade[
                [
                    "Pais",
                    "mortalidade_infantil",
                    "perfil"
                ]
            ].rename(
                columns={
                    "Pais": "País",
                    "mortalidade_infantil": "Mortalidade infantil",
                    "perfil": "Perfil de intervenção"
                }
            )

            st.dataframe(
                tabela_mortalidade,
                column_config={
                    "País": st.column_config.TextColumn(
                        "País",
                        width="medium"
                    ),
                    "Mortalidade infantil": st.column_config.NumberColumn(
                        "Mortalidade infantil",
                        format="%.2f",
                        width="medium"
                    ),
                    "Perfil de intervenção": st.column_config.TextColumn(
                        "Perfil de intervenção",
                        width="large"
                    )
                },
                width="stretch",
                hide_index=True
            )

            st.divider()

            # RANKING COMPLETO

            st.markdown(
                "#### Ranking completo de vulnerabilidade"
            )

            colunas_exibicao = [
                "Posição",
                "Pais",
                "indice_vulnerabilidade",
                "combustiveis_limpos",
                "acesso_eletricidade",
                "mortalidade_infantil",
                "perfil"
            ]

            tabela_ranking = ranking[
                colunas_exibicao
            ].rename(
                columns={
                    "Pais": "País",
                    "indice_vulnerabilidade": "Índice (0–100)",
                    "combustiveis_limpos": "Combustíveis limpos (%)",
                    "acesso_eletricidade": "Eletricidade (%)",
                    "mortalidade_infantil": "Mortalidade infantil",
                    "perfil": "Perfil de intervenção"
                }
            )

            st.dataframe(
                tabela_ranking,
                column_config={
                    "Posição": st.column_config.NumberColumn(
                        "Posição",
                        width="small"
                    ),
                    "País": st.column_config.TextColumn(
                        "País",
                        width="medium"
                    ),
                    "Índice (0–100)": st.column_config.NumberColumn(
                        "Índice (0–100)",
                        format="%.2f",
                        width="medium"
                    ),
                    "Combustíveis limpos (%)": st.column_config.NumberColumn(
                        "Combustíveis limpos (%)",
                        format="%.2f",
                        width="medium"
                    ),
                    "Eletricidade (%)": st.column_config.NumberColumn(
                        "Eletricidade (%)",
                        format="%.2f",
                        width="medium"
                    ),
                    "Mortalidade infantil": st.column_config.NumberColumn(
                        "Mortalidade infantil",
                        format="%.2f",
                        width="medium"
                    ),
                    "Perfil de intervenção": st.column_config.TextColumn(
                        "Perfil de intervenção",
                        width="large"
                    )
                },
                width="stretch",
                hide_index=True
            )

            # ----------------------------------------------------
            # EXPORTAÇÃO DO RANKING EM CSV
            # ----------------------------------------------------

            csv_ranking = tabela_ranking.to_csv(
                index=False,
                sep=";",
                decimal=",",
                float_format="%.2f"
            ).encode("utf-8-sig")

            st.download_button(
                label="📥 Baixar ranking completo (CSV)",
                data=csv_ranking,
                file_name="ranking_vulnerabilidade_2020.csv",
                mime="text/csv",
                help="Baixe o ranking completo de vulnerabilidade em formato CSV."
        )

            st.caption("""
            Fonte: Our World in Data e processamento próprio.
            Índice calculado pela média dos percentis dos três indicadores.
            Os valores são apresentados com duas casas decimais.
            A classificação é relativa à amostra com dados completos
            e não representa uma relação causal.
            """)

        else:

            st.warning(
                "Não existem observações com dados completos "
                "para os três indicadores."
            )

        st.divider()

        # ----------------------------------------------------
        # GRÁFICO DE DISPERSÃO
        # ----------------------------------------------------

        st.subheader(
            "Energia limpa e mortalidade infantil"
        )

        fig = grafico_energia_mortalidade(df_filtrado)

        if fig is not None:

            st.plotly_chart(
                fig,
                width="stretch"
            )

            st.caption(
                "Classificação K-Means baseada nos indicadores "
                "disponíveis na amostra de 2020. Associação "
                "estatística não representa necessariamente causalidade."
            )

        else:

            st.warning(
                "Não existem dados suficientes para gerar o gráfico."
            )

        st.divider()

        # ----------------------------------------------------
        # DISTRIBUIÇÃO DOS PERFIS
        # ----------------------------------------------------

        st.subheader("Distribuição dos perfis")

        distribuicao = (
            df_filtrado["perfil"]
            .value_counts()
            .rename_axis("perfil")
            .reset_index(name="quantidade")
        )

        if not distribuicao.empty:

            fig_perfis = px.bar(
                distribuicao,
                x="perfil",
                y="quantidade",
                color="perfil",
                text="quantidade",
                title="Países por perfil de intervenção"
            )

            st.plotly_chart(
                fig_perfis,
                width="stretch"
            )

        else:

            st.warning(
                "Nenhum perfil disponível para exibição."
            )

        st.divider()

        # ----------------------------------------------------
        # TABELA DE PAÍSES
        # ----------------------------------------------------

        st.subheader("Base de países")

        colunas = [
            "Pais",
            "Ano",
            "perfil",
            "combustiveis_limpos",
            "acesso_eletricidade",
            "mortalidade_infantil",
            "pib_per_capita"
        ]

        if not df_filtrado.empty:

            st.dataframe(
                df_filtrado[colunas].sort_values(
                    "mortalidade_infantil",
                    ascending=False
                ),
                column_config={
                    "combustiveis_limpos": st.column_config.NumberColumn(
                        "Combustíveis limpos (%)",
                        format="%.2f"
                    ),
                    "acesso_eletricidade": st.column_config.NumberColumn(
                        "Eletricidade (%)",
                        format="%.2f"
                    ),
                    "mortalidade_infantil": st.column_config.NumberColumn(
                        "Mortalidade infantil",
                        format="%.2f"
                    ),
                    "pib_per_capita": st.column_config.NumberColumn(
                        "PIB per capita",
                        format="%.2f"
                    )
                },
                width="stretch",
                hide_index=True
            )

        else:

            st.warning(
                "Nenhum país corresponde aos filtros selecionados."
            )

    else:

        st.warning(
            "Arquivo painel_clusters.csv não encontrado."
        )



    # ----------------------------------------------------
    # INTELIGÊNCIA PARA POLÍTICAS PÚBLICAS
    # ----------------------------------------------------

    st.markdown("---")
    st.header("Inteligência para Políticas Públicas")
    st.write(
        "Selecione um país para visualizar seu perfil de vulnerabilidade "
        "e recomendações de intervenção baseadas nos indicadores disponíveis."
    )

    if not df_clusters.empty:

        paises_disponiveis = sorted(df_clusters["Pais"].dropna().unique())

        pais_selecionado = st.selectbox(
            "Selecione um país:",
            paises_disponiveis,
            key="pais_recomendacoes"
        )

        dados_pais = df_clusters[
            df_clusters["Pais"] == pais_selecionado
        ].iloc[0]

        perfil = dados_pais["perfil"]
        combustiveis = dados_pais["combustiveis_limpos"]
        eletricidade = dados_pais["acesso_eletricidade"]
        mortalidade = dados_pais["mortalidade_infantil"]

        st.subheader(f"Diagnóstico: {pais_selecionado}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Acesso a combustíveis limpos",
            f"{combustiveis:.2f}%"
        )

        col2.metric(
            "Acesso à eletricidade",
            f"{eletricidade:.2f}%"
        )

        col3.metric(
            "Mortalidade infantil",
            f"{mortalidade:.2f}"
        )

        st.markdown(f"**Perfil de intervenção:** {perfil}")

        st.markdown("### Recomendações estratégicas")

        recomendacoes = []

        if combustiveis < 50:
            recomendacoes.append(
                "Ampliar o acesso a combustíveis limpos, com prioridade "
                "para famílias em situação de vulnerabilidade."
            )
        elif combustiveis < 90:
            recomendacoes.append(
                "Acelerar a transição para tecnologias limpas de cocção, "
                "com foco nas áreas ainda não atendidas."
            )
        else:
            recomendacoes.append(
                "Manter a cobertura de combustíveis limpos e monitorar "
                "a continuidade do acesso."
            )

        if eletricidade < 50:
            recomendacoes.append(
                "Priorizar a expansão da infraestrutura elétrica, "
                "especialmente em comunidades isoladas."
            )
        elif eletricidade < 90:
            recomendacoes.append(
                "Reduzir desigualdades territoriais no acesso à eletricidade "
                "e melhorar a continuidade do fornecimento."
            )
        else:
            recomendacoes.append(
                "Preservar a universalização do acesso e monitorar "
                "a qualidade e a confiabilidade do serviço."
            )

        if mortalidade >= 5:
            recomendacoes.append(
                "Articular políticas de energia limpa com ações de saúde "
                "materno-infantil, saneamento e atenção básica."
            )
        else:
            recomendacoes.append(
                "Manter o monitoramento dos indicadores de saúde infantil "
                "e fortalecer ações preventivas."
            )

        if perfil == "1. Alta Prioridade (Baixo Acesso)":
            recomendacoes.append(
                "Direcionar investimentos e cooperação internacional "
                "para acelerar o acesso à energia moderna."
            )
        elif perfil == "2. Em Transição":
            recomendacoes.append(
                "Acelerar a transição energética com políticas focalizadas "
                "nas populações que ainda enfrentam barreiras de acesso."
            )
        else:
            recomendacoes.append(
                "Consolidar os avanços e compartilhar boas práticas "
                "de implementação de políticas energéticas."
            )

        for i, recomendacao in enumerate(recomendacoes, start=1):
            st.markdown(f"**{i}.** {recomendacao}")

        st.caption(
            "As recomendações são sugestões indicativas baseadas nos "
            "indicadores observados. As associações não demonstram "
            "causalidade e devem ser complementadas por análises "
            "territoriais e institucionais."
        )

    else:
        st.warning("Não há dados disponíveis para gerar recomendações.")

# ============================================================
# ABA 2 - EVOLUÇÃO HISTÓRICA
# ============================================================



with tab2:

    st.header("Evolução histórica")

    if not df_historico.empty:

        paises = sorted(
            df_historico["Entity"].dropna().unique()
        )

        pais_selecionado = st.selectbox(
            "Selecione um país ou território",
            paises,
            index=(
                paises.index("Brazil")
                if "Brazil" in paises
                else 0
            )
        )

        indicadores = {
            "Mortalidade infantil": "mortalidade_infantil",
            "Combustíveis limpos (%)": "combustiveis_limpos",
            "Acesso à eletricidade (%)": "acesso_eletricidade",
            "PIB per capita": "pib_per_capita"
        }

        indicador_selecionado = st.selectbox(
            "Selecione o indicador",
            list(indicadores.keys())
        )

        coluna_indicador = indicadores[
            indicador_selecionado
        ]

        df_pais = df_historico[
            df_historico["Entity"] == pais_selecionado
        ].copy()

        df_pais = df_pais.dropna(
            subset=[coluna_indicador]
        ).sort_values("Year")

        if not df_pais.empty:

            fig_historico = px.line(
                df_pais,
                x="Year",
                y=coluna_indicador,
                markers=True,
                title=(
                    f"{indicador_selecionado} — "
                    f"{pais_selecionado}"
                ),
                labels={
                    "Year": "Ano",
                    coluna_indicador: indicador_selecionado
                }
            )

            st.plotly_chart(
                fig_historico,
                width="stretch"
            )

            st.caption(
                "Fonte: Our World in Data. A disponibilidade de dados "
                "varia entre indicadores, países e períodos."
            )

        else:

            st.warning(
                "Não existem dados disponíveis para essa combinação."
            )

    else:

        st.warning(
            "Base histórica não encontrada."
        )


# ============================================================
# ABA 3 - COMPARAÇÃO ENTRE PAÍSES
# ============================================================

with tab3:

    st.header("Comparação internacional")

    if not df_historico.empty:

        paises = sorted(
            df_historico["Entity"].dropna().unique()
        )

        col1, col2 = st.columns(2)

        with col1:

            pais1 = st.selectbox(
                "Primeiro país",
                paises,
                index=(
                    paises.index("Brazil")
                    if "Brazil" in paises
                    else 0
                ),
                key="pais_comparacao_1"
            )

        with col2:

            indice_pais2 = (
                paises.index("Zimbabwe")
                if "Zimbabwe" in paises
                else min(1, len(paises) - 1)
            )

            pais2 = st.selectbox(
                "Segundo país",
                paises,
                index=indice_pais2,
                key="pais_comparacao_2"
            )

        indicadores_comparacao = {
            "Mortalidade infantil": "mortalidade_infantil",
            "Combustíveis limpos (%)": "combustiveis_limpos",
            "Acesso à eletricidade (%)": "acesso_eletricidade",
            "PIB per capita": "pib_per_capita"
        }

        indicador = st.selectbox(
            "Indicador para comparação",
            list(indicadores_comparacao.keys()),
            key="indicador_comparacao"
        )

        coluna = indicadores_comparacao[indicador]

        dados = df_historico[
            df_historico["Entity"].isin([pais1, pais2])
        ][
            ["Entity", "Year", coluna]
        ].dropna()

        if not dados.empty:

            fig_comparacao = px.line(
                dados,
                x="Year",
                y=coluna,
                color="Entity",
                markers=True,
                title=f"Comparação de {indicador}",
                labels={
                    "Year": "Ano",
                    coluna: indicador,
                    "Entity": "País ou território"
                }
            )

            st.plotly_chart(
                fig_comparacao,
                width="stretch"
            )

            st.caption(
                "Fonte: Our World in Data. As séries podem apresentar "
                "períodos diferentes de disponibilidade de dados."
            )

        else:

            st.warning(
                "Não existem dados para os países e indicador selecionados."
            )

    else:

        st.warning(
            "Base histórica não encontrada."
        )


# ============================================================
# ABA 4 - PREVISÃO LSTM

# ============================================================
# ABA 4 - PREVISAO LSTM
# ============================================================

with tab4:

    import json

    st.header("Previsao da mortalidade de menores de cinco anos")

    st.markdown("""
    Projecoes produzidas por uma rede neural LSTM global,
    utilizando series historicas de diferentes paises.
    """)

    pasta_dados = BASE / "data" / "processed"

    arquivo_backtest = pasta_dados / "lstm_backtest_recursivo_2020_2024.csv"
    arquivo_anual = pasta_dados / "lstm_backtest_metricas_anuais.csv"
    arquivo_sensibilidade = pasta_dados / "lstm_sensibilidade.csv"
    arquivo_previsoes = pasta_dados / "lstm_previsoes_2025_2029.csv"

    arquivos = [
        arquivo_backtest,
        arquivo_anual,
        arquivo_sensibilidade,
        arquivo_previsoes
    ]

    if all(a.exists() for a in arquivos):

        backtest = pd.read_csv(arquivo_backtest)
        anual = pd.read_csv(arquivo_anual)
        sensibilidade = pd.read_csv(arquivo_sensibilidade)
        previsoes = pd.read_csv(arquivo_previsoes)

        # ====================================================
        # 1. INDICADORES GERAIS
        # ====================================================

        erro_lstm = backtest["valor_real"] - backtest["previsao_lstm"]
        erro_base = backtest["valor_real"] - backtest["previsao_baseline"]

        mae_lstm = erro_lstm.abs().mean()
        mae_base = erro_base.abs().mean()

        rmse_lstm = (erro_lstm.pow(2).mean()) ** 0.5
        rmse_base = (erro_base.pow(2).mean()) ** 0.5

        st.subheader("Desempenho geral do modelo")

        st.caption(
            "Backtest recursivo de 2020 a 2024, com origem temporal em 2019."
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("MAE - LSTM", f"{mae_lstm:.4f}")
        c2.metric("MAE - Persistencia", f"{mae_base:.4f}")
        c3.metric("RMSE - LSTM", f"{rmse_lstm:.4f}")
        c4.metric("RMSE - Persistencia", f"{rmse_base:.4f}")

        st.caption(
            "Metricas em pontos percentuais. Valores menores representam "
            "menor erro de previsao."
        )

        st.divider()

        # ====================================================
        # 2. DESEMPENHO ANUAL
        # ====================================================

        st.subheader("Desempenho anual: 2020-2024")

        col1, col2 = st.columns(2)

        with col1:

            fig_mae = px.bar(
                anual,
                x="ano",
                y=["mae_lstm", "mae_baseline"],
                barmode="group",
                title="Erro absoluto medio (MAE)",
                labels={
                    "ano": "Ano",
                    "value": "MAE",
                    "variable": "Modelo"
                }
            )

            fig_mae.for_each_trace(
                lambda t: t.update(
                    name="LSTM" if t.name == "mae_lstm"
                    else "Persistencia"
                )
            )

            st.plotly_chart(fig_mae, width="stretch")

        with col2:

            fig_rmse = px.bar(
                anual,
                x="ano",
                y=["rmse_lstm", "rmse_baseline"],
                barmode="group",
                title="Raiz do erro quadratico medio (RMSE)",
                labels={
                    "ano": "Ano",
                    "value": "RMSE",
                    "variable": "Modelo"
                }
            )

            fig_rmse.for_each_trace(
                lambda t: t.update(
                    name="LSTM" if t.name == "rmse_lstm"
                    else "Persistencia"
                )
            )

            st.plotly_chart(fig_rmse, width="stretch")

        st.dataframe(
            anual.rename(columns={
                "ano": "Ano",
                "registros": "Observacoes",
                "paises": "Paises",
                "mae_lstm": "MAE LSTM",
                "rmse_lstm": "RMSE LSTM",
                "mae_baseline": "MAE Persistencia",
                "rmse_baseline": "RMSE Persistencia"
            }).round(4),
            hide_index=True,
            width="stretch"
        )

        st.divider()

        # ====================================================
        # 3. ANALISE DE SENSIBILIDADE
        # ====================================================

        st.subheader("Analise de sensibilidade")

        st.markdown("""
        Comparacao entre a amostra completa e cenarios que retiram
        progressivamente paises com erros elevados. A amostra completa
        permanece como resultado principal.
        """)

        tabela_sensibilidade = sensibilidade.rename(columns={
            "cenario": "Cenario",
            "paises": "Paises",
            "observacoes": "Observacoes",
            "MAE_LSTM": "MAE LSTM",
            "RMSE_LSTM": "RMSE LSTM",
            "MAE_baseline": "MAE Persistencia",
            "RMSE_baseline": "RMSE Persistencia"
        })

        st.dataframe(
            tabela_sensibilidade.round(4),
            hide_index=True,
            width="stretch"
        )

        col1, col2 = st.columns(2)

        with col1:

            fig_sens_mae = px.bar(
                sensibilidade,
                x="cenario",
                y=["MAE_LSTM", "MAE_baseline"],
                barmode="group",
                title="Sensibilidade do MAE",
                labels={
                    "cenario": "Cenario",
                    "value": "MAE",
                    "variable": "Modelo"
                }
            )

            fig_sens_mae.update_layout(xaxis_tickangle=-25)

            fig_sens_mae.for_each_trace(
                lambda t: t.update(
                    name="LSTM" if t.name == "MAE_LSTM"
                    else "Persistencia"
                )
            )

            st.plotly_chart(fig_sens_mae, width="stretch")

        with col2:

            fig_sens_rmse = px.bar(
                sensibilidade,
                x="cenario",
                y=["RMSE_LSTM", "RMSE_baseline"],
                barmode="group",
                title="Sensibilidade do RMSE",
                labels={
                    "cenario": "Cenario",
                    "value": "RMSE",
                    "variable": "Modelo"
                }
            )

            fig_sens_rmse.update_layout(xaxis_tickangle=-25)

            fig_sens_rmse.for_each_trace(
                lambda t: t.update(
                    name="LSTM" if t.name == "RMSE_LSTM"
                    else "Persistencia"
                )
            )

            st.plotly_chart(fig_sens_rmse, width="stretch")

        st.divider()

        # ====================================================
        # 4. PROJECOES 2025-2029
        # ====================================================

        st.subheader("Projecoes para 2025-2029")

        st.markdown("""
        Selecione um pais para consultar a trajetoria estimada.
        As previsoes sao recursivas: cada estimativa alimenta
        a janela utilizada para calcular o ano seguinte.
        """)

        paises = sorted(
            previsoes["pais"].dropna().unique().tolist()
        )

        if paises:

            indice_brasil = paises.index("Brazil") if "Brazil" in paises else 0

            pais = st.selectbox(
                "Selecione o pais",
                paises,
                index=indice_brasil
            )

            dados_pais = (
                previsoes[previsoes["pais"] == pais]
                .sort_values("ano")
            )

            fig_projecao = px.line(
                dados_pais,
                x="ano",
                y="mortalidade_prevista",
                markers=True,
                title=f"Projecao para {pais}",
                labels={
                    "ano": "Ano",
                    "mortalidade_prevista": "Mortalidade prevista (%)"
                }
            )

            st.plotly_chart(fig_projecao, width="stretch")

            st.dataframe(
                dados_pais.rename(columns={
                    "pais": "Pais",
                    "ano": "Ano",
                    "mortalidade_prevista": "Mortalidade prevista (%)"
                }).round(3),
                hide_index=True,
                width="stretch"
            )

        else:

            st.warning("Nao existem projecoes disponiveis.")

        st.divider()

        # ====================================================
        # 5. METODOLOGIA E LIMITACOES
        # ====================================================

        st.subheader("Metodologia e limitacoes")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("""
            **Configuracao**

            - Modelo: LSTM global univariada.
            - Janela historica: 5 anos.
            - Treinamento original: ate 2014.
            - Validacao original: 2015-2019.
            - Backtest recursivo: 2020-2024.
            - Projecao: 2025-2029.
            """)

        with col2:

            st.markdown("""
            **Limitacoes**

            - Backtest com uma unica origem temporal: 2019.
            - Desempenho heterogeneo entre paises.
            - Sensibilidade a trajetorias muito volateis.
            - Ausencia de intervalos de previsao.
            - Resultados nao demonstram causalidade.
            """)

        st.info(
            "As projecoes sao estimativas exploratorias. "
            "Nao representam previsoes oficiais nem resultados garantidos."
        )

    else:

        st.warning(
            "Arquivos da avaliacao LSTM nao encontrados. "
            "Verifique os arquivos na pasta data/processed."
        )
