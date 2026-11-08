
import plotly.express as px
import pandas as pd


# ============================================================
# GRÁFICO DE DISPERSÃO
# ============================================================

def grafico_energia_mortalidade(df):
    """
    Cria gráfico de dispersão entre acesso a combustíveis
    limpos e mortalidade infantil.
    """

    colunas = [
        "combustiveis_limpos",
        "mortalidade_infantil",
        "perfil"
    ]

    dados = df.dropna(subset=colunas).copy()

    if dados.empty:
        return None

    fig = px.scatter(
        dados,
        x="combustiveis_limpos",
        y="mortalidade_infantil",
        color="perfil",
        hover_name="Pais",
        size="pib_per_capita",
        size_max=25,
        labels={
            "combustiveis_limpos": "Acesso a combustíveis limpos (%)",
            "mortalidade_infantil": "Mortalidade infantil (por 1.000)",
            "perfil": "Perfil de intervenção",
            "pib_per_capita": "PIB per capita"
        },
        title="Distribuição dos países por perfil"
    )

    fig.update_layout(
        font=dict(size=11),
        title=dict(font=dict(size=17)),
        legend=dict(
            font=dict(size=10),
            title=dict(font=dict(size=11))
        ),
        hoverlabel=dict(
            font=dict(size=11),
            namelength=-1
        ),
        margin=dict(l=20, r=20, t=65, b=20),
        autosize=True
    )

    return fig


# ============================================================
# MAPA-MÚNDI INTERATIVO
# ============================================================

def grafico_mapa_mundi(df):
    """
    Cria mapa-múndi interativo com os três perfis
    de intervenção energética.
    """

    colunas = [
        "Pais",
        "perfil",
        "combustiveis_limpos",
        "acesso_eletricidade",
        "mortalidade_infantil",
        "pib_per_capita"
    ]

    dados = df.dropna(
        subset=["Pais", "perfil"]
    )[colunas].copy()

    if dados.empty:
        return None

    cores_perfis = {
        "1. Alta Prioridade (Baixo Acesso)": "#D9534F",
        "2. Em Transição": "#F0AD4E",
        "3. Consolidado (Alto Acesso)": "#28A745"
    }

    fig = px.choropleth(
        dados,
        locations="Pais",
        locationmode="country names",
        color="perfil",
        hover_name="Pais",
        hover_data={
            "perfil": True,
            "combustiveis_limpos": ":.1f",
            "acesso_eletricidade": ":.1f",
            "mortalidade_infantil": ":.2f",
            "pib_per_capita": ":,.2f",
            "Pais": False
        },
        color_discrete_map=cores_perfis,
        category_orders={
            "perfil": [
                "1. Alta Prioridade (Baixo Acesso)",
                "2. Em Transição",
                "3. Consolidado (Alto Acesso)"
            ]
        },
        labels={
            "perfil": "Perfil de intervenção",
            "combustiveis_limpos": "Combustíveis limpos (%)",
            "acesso_eletricidade": "Acesso à eletricidade (%)",
            "mortalidade_infantil": "Mortalidade infantil (por 1.000)",
            "pib_per_capita": "PIB per capita (US$)"
        },
        title="Mapa global dos perfis de intervenção energética"
    )

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#263238",
        showocean=True,
        oceancolor="#111827",
        showcountries=True,
        countrycolor="#64748B",
        showcoastlines=True,
        coastlinecolor="#64748B",
        showframe=False
    )

    fig.update_layout(
        height=650,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Arial",
            size=11,
            color="#FFFFFF"
        ),
        title=dict(font=dict(size=17)),
        legend=dict(
            font=dict(size=10),
            title=dict(font=dict(size=11))
        ),
        hoverlabel=dict(
            font=dict(size=11),
            namelength=-1,
            align="left"
        ),
        legend_title_text="Perfil de intervenção",
        margin=dict(l=10, r=10, t=65, b=10)
    )

    return fig
