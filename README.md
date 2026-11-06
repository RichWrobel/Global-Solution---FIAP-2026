# 🌍 O Preço Oculto da Energia Suja

## 👨‍🎓 Alunos 

👤Douglas Felicio 
✅RM: 572312

👤Gabriel Kaique 
✅RM: 572500

👤Richard Wrobel dos Santos 
✅RM: 573998

## 📌 O Problema
A poluição do ar doméstico (causada pelo uso de lenha e carvão para cozinhar) é uma das principais causas de morte evitável em crianças no mundo. Este projeto investiga como o acesso a combustíveis limpos impacta a mortalidade infantil.

## 🛠️ Tecnologias e Fases do Projeto (FIAP)
O projeto foi construído integrando todas as competências do curso de IA:
- **Python & Pandas (Fase 3):** Coleta e limpeza de dados abertos do Our World in Data (OWID).
- **Machine Learning (Fase 4/5):** Regressão Linear, Random Forest e K-Means (Clusterização).
- **Deep Learning (Fase 6/7):** Redes Neurais Recorrentes (LSTM) para previsão e CNNs (MobileNetV2) para Visão Computacional.
- **Cloud & IoT (Fase 5/6/7):** Arquitetura proposta com AWS (Lambda/EC2) e sensores ESP32.
- **Frontend:** Dashboard interativo com Streamlit e Plotly.

## 📊 Resultados Principais
1. A relação entre acesso e mortalidade **não é linear**. O maior ganho marginal em vidas salvas ocorre na faixa de 0 a 30% de acesso.
2. O K-Means agrupou os países em 3 perfis, identificando os locais de **Alta Prioridade** para intervenções de políticas públicas.
3. A rede LSTM projetou a estabilização da mortalidade no Brasil para os próximos 5 anos.

## 🚀 Como rodar o projeto
1. Clone o repositório.
2. Instale as dependências: `pip install -r requirements.txt`
3. Baixe os dados: `python src/baixar_owid.py`
4. Rode o dashboard: `streamlit run src/app.py`

