# Explorando a Rede Científica de Computação no Brasil 🇧🇷

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-em%20desenvolvimento-orange)](#)

Este repositório contém o ecossistema de dados e análises para o mapeamento da Ciência da Computação no Brasil. Através da integração de dados da **CAPES**, **Plataforma Lattes** e **OpenAlex**, o projeto aplica Processamento de Linguagem Natural (**SciBERT**) e **Ciência de Redes** para revelar padrões de colaboração e impacto na comunidade científica nacional.

---

## 📖 Visão Geral

O projeto transforma o perfil de publicações de pesquisadores em **embeddings semânticos** (representações vetoriais densas). Este "mapa" permite investigar:

* **Comunidades Emergentes:** Identificação de clusters por afinidade temática real, além das divisões administrativas.
* **Dinâmica de Colaboração:** Análise se a proximidade científica supera barreiras geográficas ou institucionais.
* **Topologia de Impacto:** Localização de pesquisadores de alto impacto (Bolsistas PQ) na estrutura da rede.
* **Papel Institucional:** O impacto de programas de excelência (notas 6 e 7 CAPES) na conectividade da rede nacional.

---

## ✨ Destaques Técnicos

* **Pipeline de Dados:** Coleta automatizada via API OpenAlex com tratamento de dados brutos.
* **NLP Científico:** Uso do modelo **SciBERT** (treinado em 1.14M de artigos científicos) para garantir precisão semântica.
* **Análise de Redes:** Métricas de centralidade e agrupamento aplicadas a grafos de coautoria.
* **Reprodutibilidade:** Estrutura modular que permite a atualização dos dados com esforço mínimo.

---

## 📂 Estrutura do Projeto

    /scisci-bert/
    ├── data/
    │   ├── raw/             # Dados brutos (ex: researchers.csv inicial)
    │   └── processed/       # Datasets limpos e embeddings gerados
    ├── notebooks/           # Jupyter Notebooks para análise e visualização
    │   ├── 01_Semantic_Space_Validation.ipynb
    │   └── 02_Network_and_Collaboration_Analysis.ipynb
    ├── src/                 # Código-fonte modular
    │   ├── data_collection/ # Scraping e consumo de APIs (OpenAlex)
    │   ├── feature_engineering/ # Geração de embeddings (SciBERT)
    │   ├── analysis/        # Algoritmos de clustering e redes
    │   └── visualization/   # Scripts de plotagem (Plotly/Seaborn)
    ├── reports/
    │   └── figures/         # Gráficos exportados para a monografia
    ├── .env_template        # Configurações de ambiente
    └── requirements.txt     # Dependências do projeto

---

## 🚀 Como Executar

### 1. Preparação do Ambiente

Clone o repositório:
    git clone https://github.com/jhonatangs/scisci-bert.git
    cd scisci-bert

Crie o ambiente virtual:
    python -m venv venv

Ative o ambiente no Linux/macOS:
    source venv/bin/activate

Ou ative o ambiente no Windows:
    .\venv\Scripts\activate

Instale as dependências:
    pip install -r requirements.txt

### 2. Configuração de Variáveis

Copie o arquivo `.env_template` para `.env` e insira seu e-mail para acesso "polite" à API OpenAlex:

    cp .env_template .env

### 3. Preparação dos Dados

Coloque seu arquivo inicial em `data/raw/researchers.csv`. Ele deve seguir este formato:

| name | lattes_id | institution |
| :--- | :--- | :--- |
| Nome do Pesquisador | 123456789... | UFOP |

### 4. Execução do Pipeline

Siga a ordem lógica para processamento completo:

1.  **Coleta:** `python -m src.data_collection.main`
2.  **Embeddings:** `python -m src.feature_engineering.main` 
    * *Nota: Recomendado uso de GPU com suporte a CUDA.*
3.  **Análise:** Execute os notebooks em `/notebooks` seguindo a numeração.

---

## 🔬 Questões de Pesquisa (RQs)

* **RQ 1.1:** Validação da correspondência entre clusters semânticos e áreas formais (SBC).
* **RQ 1.2:** Diferenciação espacial de pesquisadores de alto impacto no mapa semântico.
* **RQ 2.1:** Correlação entre proximidade temática e probabilidade de coautoria.
* **RQ 2.3:** Análise de Programas de Pós-Graduação (PPGs) como hubs de conhecimento.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Processamento:** Pandas, NumPy, Scikit-learn, PyTorch
* **Embeddings:** Hugging Face Transformers (SciBERT)
* **Gráficos & Redes:** NetworkX, Plotly, UMAP, Seaborn
* **Fontes de Dados:** OpenAlex API (via Pyalex)

---

## 📄 Licença e Citação

Distribuído sob a Licença MIT. Veja `LICENSE` para mais informações.

**Citação Sugerida:**
> SOUZA, J. G. **Explorando a Rede Científica de Computação no Brasil: Abordagem Integrada com BERT, Análise de Agrupamento e OpenAlex**. Monografia (Bacharelado em Ciência da Computação) - Universidade Federal de Ouro Preto, 2025.
