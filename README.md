# 🌍 O PREÇO OCULTO DA ENERGIA SUJA
### ⚡ Energia limpa, desigualdade energética e saúde infantil

<p align="center">
  **FIAP GLOBAL SOLUTION 2026.2**  
  **EQUIPE 55**
</p>

> 💡 **E se o acesso à energia limpa também fosse uma questão de saúde pública?**

Um projeto de ciência de dados que investiga a relação entre o acesso a combustíveis limpos e a mortalidade de menores de cinco anos, explorando desigualdades econômicas e territoriais por meio de análise estatística, econometria e inteligência artificial.

---

## 🔎 O projeto em números

| 🌎 Países no ranking | 📊 Painel histórico | 🤖 Países no backtest LSTM | 🔮 Horizonte de projeção |
|:---:|:---:|:---:|:---:|
| **142** | **18.842 registros** | **199** | **2025–2029** |

---

## 🎯 01. O desafio

O acesso à energia é um dos pilares do desenvolvimento humano. Apesar dos avanços tecnológicos, milhões de pessoas ainda dependem de combustíveis poluentes para preparar alimentos.

A exposição à poluição do ar doméstico representa um desafio de saúde pública, especialmente para populações em situação de vulnerabilidade.

Mas será que conseguimos identificar uma relação entre as desigualdades energéticas e os indicadores de saúde infantil?

É a partir dessa questão que desenvolvemos este projeto.

### ❓ Nossa pergunta de pesquisa

**Como o acesso a combustíveis limpos para cozinhar se relaciona com a mortalidade de menores de cinco anos, considerando as desigualdades socioeconômicas e territoriais entre países?**

### 🎯 Nosso objetivo

Utilizar dados internacionais e ferramentas analíticas para identificar padrões, comparar países, construir indicadores de vulnerabilidade e explorar cenários relacionados à transição energética e à saúde infantil.

---

## 🧭 02. O que queremos investigar?

Nosso estudo integra cinco dimensões:

| Dimensão | Indicador | O que buscamos compreender |
|---|---|---|
| 🔥 Energia limpa | Acesso a combustíveis limpos | Desigualdade no acesso a tecnologias de cocção limpa |
| ⚡ Infraestrutura | Acesso à eletricidade | Condições de infraestrutura energética |
| 🏥 Saúde | Mortalidade de menores de cinco anos | Diferenças nos indicadores de saúde infantil |
| 💰 Economia | PIB per capita | Diferenças nas condições econômicas |
| 🎓 Educação | Escolaridade | Desigualdades educacionais |

> **Nota importante:** o indicador de mortalidade utilizado representa a proporção de crianças que morrem antes de completar cinco anos, expressa por 100 nascidos vivos (%). Não se trata exclusivamente da mortalidade durante o primeiro ano de vida.

---

## 🧪 03. Como desenvolvemos a análise?

O projeto combina diferentes abordagens para explorar o problema sob perspectivas complementares.

### 📊 Análise exploratória e integração de dados

- Integração de indicadores internacionais.
- Organização de dados por país e ano.
- Verificação de consistência e duplicidades.
- Análise de valores ausentes.
- Investigação de padrões históricos e territoriais.

### 🌎 Classificação territorial

Organizamos uma classificação comparativa de 142 países em três perfis:

| Perfil | Significado |
|---|---|
| 🟢 **Consolidados** | Condições relativamente mais favoráveis de acesso |
| 🟡 **Transição** | Situação intermediária |
| 🔴 **Alta prioridade** | Condições relativamente mais críticas |

### 🏆 Ranking de vulnerabilidade

Construímos um índice comparativo para 142 países, considerando o ano de 2020.

O índice combina três dimensões:

1. Menor acesso a combustíveis limpos.
2. Menor acesso à eletricidade.
3. Maior mortalidade de menores de cinco anos.

As dimensões são transformadas em posições percentílicas de vulnerabilidade e agregadas por média.

O resultado permite comparar a vulnerabilidade relativa entre os países da amostra.

### 📐 Análise econométrica

Utilizamos diferentes especificações para investigar associações entre os indicadores:

- **Efeitos fixos:** controle de características específicas dos países e de fatores comuns a cada ano.
- **Mean Group (MG):** estimação de relações individuais por país e agregação dos coeficientes.
- **Common Correlated Effects Mean Group (CCEMG):** inclusão de médias transversais para considerar fatores comuns entre países.
- **Análises de sensibilidade e bootstrap:** investigação da estabilidade das estimativas.

### 🤖 Inteligência artificial e aprendizado de máquina

Implementamos uma rede neural **Long Short-Term Memory (LSTM)** para explorar padrões temporais da mortalidade de menores de cinco anos.

A arquitetura utiliza sequências históricas de cinco anos e inclui:

| Componente | Configuração |
|---|---|
| Modelo | LSTM |
| Janela temporal | 5 anos |
| Camada recorrente | 32 unidades |
| Camada densa | 16 unidades |
| Otimizador | Adam |
| Função de perda | Huber |
| Projeções | 2025–2029 |

A implementação inclui normalização, transformação logarítmica, treinamento, avaliação temporal e geração recursiva de projeções.

---

## 📈 04. O que encontramos?

### 🏆 Países com maior vulnerabilidade relativa

O ranking de 2020 reúne 142 países. Entre os dez primeiros colocados estão:

| Posição | País |
|:---:|---|
| 1º | 🇨🇫 República Centro-Africana |
| 2º | 🇸🇱 Serra Leoa |
| 3º | 🇳🇪 Níger |
| 4º | 🇱🇷 Libéria |
| 5º | 🇨🇩 República Democrática do Congo |
| 6º | 🇧🇮 Burundi |
| 7º | 🇲🇱 Mali |
| 8º | 🇲🇼 Malawi |
| 9º | 🇧🇯 Benim |
| 10º | 🇲🇿 Moçambique |

Essas posições representam a combinação relativa das dimensões utilizadas no índice. Não constituem uma classificação oficial nem um diagnóstico completo das condições de cada país.

### 📐 O que indicaram os modelos econométricos?

Os modelos com efeitos fixos apresentaram coeficientes negativos para o acesso a combustíveis limpos, mas sem significância estatística convencional.

As análises MG e CCEMG também não identificaram evidência estatística robusta de uma associação negativa média nas especificações examinadas.

Isso reforça a importância de considerar a heterogeneidade entre países e os diversos fatores que influenciam simultaneamente as condições de energia e saúde.

### 🤖 Como a LSTM se comportou?

Realizamos um backtest recursivo entre 2020 e 2024, utilizando:

- 199 países;
- 995 observações;
- Dados de treinamento disponíveis até 2019;
- Comparação com um baseline de persistência.

**Resultados agregados:**

| Métrica | LSTM | Baseline |
|---|---:|---:|
| MAE | **0,2161** | 0,2935 |
| RMSE | **0,9719** | 1,0999 |

<details>
<summary>📊 <strong>Entenda os resultados</strong></summary>

No resultado agregado, a LSTM apresentou:

- Aproximadamente 26,4% menos erro absoluto médio.
- Aproximadamente 11,6% menos raiz do erro quadrático médio.

Entretanto, o desempenho variou entre anos e países. Em 2022, por exemplo, o RMSE da LSTM foi superior ao do baseline, evidenciando a influência de observações extremas.

O backtest utiliza uma única origem temporal de previsão, portanto seus resultados devem ser interpretados como exploratórios.

</details>

### 🔮 E o futuro?

Com os dados disponíveis até 2024, treinamos um modelo final para produzir projeções recursivas de 2025 a 2029.

Essas projeções permitem explorar possíveis trajetórias dos indicadores, mas não devem ser confundidas com previsões oficiais ou impactos comprovados de políticas públicas.

---

## 💻 05. Conheça o dashboard

O projeto conta com um dashboard interativo desenvolvido com **Streamlit e Plotly**.

Nele, o usuário pode explorar os indicadores, comparar países e visualizar os resultados das análises.

**Principais funcionalidades:**

- 📊 Indicadores gerais e informações do painel.
- 🌎 Comparações territoriais.
- 📈 Evolução histórica dos indicadores.
- 🔥 Perfis de acesso energético.
- 🏆 Ranking de vulnerabilidade.
- 📐 Resultados econométricos.
- 🤖 Avaliação do modelo LSTM.
- 🔮 Projeções por país para 2025–2029.

> O dashboard foi desenvolvido para tornar a análise mais acessível, interativa e compreensível, aproximando os resultados técnicos de uma experiência de exploração de dados.

---

## 🛠️ 06. Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| 🐍 **Python** | Linguagem principal |
| 🐼 **Pandas** | Manipulação e integração de dados |
| 🔢 **NumPy** | Operações numéricas |
| 📊 **Matplotlib** | Visualização de dados |
| 📈 **Plotly** | Gráficos interativos |
| 🖥️ **Streamlit** | Desenvolvimento do dashboard |
| 📐 **Statsmodels** | Modelagem econométrica |
| 🤖 **Scikit-learn** | Preparação e avaliação de modelos |
| 🧠 **TensorFlow / Keras** | Implementação da rede neural LSTM |

---

## 📁 07. Organização do projeto

```text
Projeto Final/
│
├── data/
│   ├── raw/
│   │   └── Bases originais e metadados
│   │
│   └── processed/
│       └── Bases integradas e resultados
│
├── models/
│   └── lstm_global.keras
│
├── src/
│   ├── FrontEnd/
│   │   └── Componentes de visualização
│   │
│   ├── modelos/
│   │   └── Scripts de modelagem
│   │
│   ├── diagnóstico/
│   │   └── Rotinas de obtenção e diagnóstico
│   │
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 08. Como executar o projeto

### Requisitos

- Python instalado.
- Git, caso o projeto seja clonado.
- Dependências disponíveis em `requirements.txt`.

### 1. Clone o repositório

```bash
git clone URL_DO_REPOSITORIO
cd NOME_DO_REPOSITORIO
```

Substitua os valores acima pelo endereço e nome efetivos do repositório.

### 2. Crie o ambiente virtual

```powershell
python -m venv .venv
```

### 3. Ative o ambiente

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Instale as dependências

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Execute o dashboard

```powershell
python -m streamlit run .\src\app.py
```

Acesse no navegador:

**http://localhost:8501**

---

## ⚠️ 09. Disclaimer — Escopo e limitações

Este projeto foi desenvolvido prioritariamente no contexto acadêmico da **FIAP Global Solution 2026.2**. Seu objetivo é aplicar ferramentas de ciência de dados, Inteligência Artificial Generativa (IAG), aprendizado de máquina e análise estatística a um problema de relevância social.

As análises econométricas, os modelos preditivos e as variáveis utilizadas devem ser interpretados com cautela. Os resultados não constituem, isoladamente, evidências definitivas para a formulação, implementação ou avaliação de políticas públicas.

Nosso objetivo é explorar como ferramentas de inteligência artificial e análise de dados podem auxiliar na identificação de padrões, na elaboração de cenários e na estimativa de possíveis impactos de políticas públicas sobre as populações estudadas.

### O que precisamos considerar?

- **Quantidade de variáveis:** um conjunto limitado de indicadores pode deixar de contemplar fatores relevantes e introduzir vieses.
- **Complexidade social:** saúde, energia e desenvolvimento são influenciados por condições econômicas, sociais, ambientais, demográficas e institucionais.
- **Limitações dos modelos:** os resultados dependem das bases de dados, das variáveis selecionadas e das premissas adotadas.
- **Causalidade:** associações estatísticas e previsões não demonstram, por si só, relações de causa e efeito.
- **Generalização:** padrões identificados em determinados países ou períodos podem não se reproduzir em outros contextos.

**Uma elevada taxa de mortalidade de menores de cinco anos em determinada região não pode ser atribuída exclusivamente à falta de acesso à energia limpa ou renovável.**

Essa condição pode estar relacionada a diversos outros fatores, como pobreza, saneamento básico, acesso aos serviços de saúde, nutrição, escolaridade, condições ambientais e infraestrutura.

> **Nossa posição:** os resultados devem ser compreendidos como instrumentos exploratórios de análise e apoio à tomada de decisão, e não como diagnósticos definitivos ou recomendações automáticas de políticas públicas.

Uma avaliação efetiva exige estudos complementares, variáveis adicionais, validação empírica e consideração das particularidades de cada população e território.

---

## 🌱 10. Considerações finais

A transição energética não é apenas uma questão de infraestrutura. Ela também está relacionada ao desenvolvimento humano, às condições de vida e aos desafios de saúde pública.

Por meio da integração de dados internacionais, modelos estatísticos e ferramentas de inteligência artificial, buscamos construir uma perspectiva analítica sobre essas desigualdades.

O projeto demonstra como a ciência de dados pode contribuir para identificar padrões, comparar realidades e explorar cenários, sem ignorar as limitações e a complexidade dos fenômenos sociais.

**Mais do que prever números, queremos utilizar dados e tecnologia para fazer perguntas melhores sobre problemas reais.**

---

## 📚 11. Fontes de dados

- 🌐 [Our World in Data](https://ourworldindata.org/)
- 🏥 [UN Inter-agency Group for Child Mortality Estimation — UN IGME](https://childmortality.org/)
- 🌍 [Gapminder](https://www.gapminder.org/)

Consulte os metadados e arquivos de origem incluídos no projeto para informações específicas sobre os indicadores, períodos e fontes utilizados.

---

<p align="center">

### 🌍 FIAP GLOBAL SOLUTION 2026.2

**EQUIPE 55**

*O Preço Oculto da Energia Suja*

⚡ Energia limpa • 📊 Ciência de dados • 🤖 Inteligência artificial • 🏥 Saúde infantil

</p>