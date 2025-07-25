# Explorando a Rede Científica de Computação no Brasil

Este repositório contém o código e as análises para o projeto de pesquisa que mapeia o cenário da ciência da computação no Brasil usando dados da OpenAlex e técnicas de Processamento de Linguagem Natural.

## Visão Geral do Projeto

Breve descrição do objetivo: construir um espaço semântico de pesquisadores para entender padrões de colaboração, impacto e estrutura da comunidade.

## Estrutura do Repositório

*   `data/`: Contém os dados brutos, processados e intermediários.
    *   `raw/`: Dados brutos, intocáveis (e.g., o CSV inicial).
    *   `processed/`: Dados limpos e enriquecidos (e.g., parquet, HDF5).
*   `notebooks/`: Notebooks para análise exploratória e visualização.
*   `src/`: Código-fonte principal do projeto.
    *   `data_collection/`: Módulos para interagir com OpenAlex.
    *   `feature_engineering/`: Módulos para gerar embeddings e outras features.
    *   `analysis/`: Módulos para clustering, análise de redes, etc.
    *   `visualization/`: Funções de plotagem reutilizáveis.
    *   `utils/`: Funções auxiliares, configurações, etc.
*   `reports/`: Contém relatórios e figuras geradas.
    *   `figures/`: Gráficos e visualizações gerados para o relatório final.
*   `env_template`: Template para variáveis de ambiente. **RENOMEIE ESTE ARQUIVO PARA .env**
*   `requirements.txt`: Dependências do projeto.
*   `README.md`: Documentação principal.

## Como Reproduzir os Resultados

Guia passo a passo detalhado:

1.  **Clonar o Repositório:**
    ```bash
    git clone [URL_DO_REPOSITORIO]
    cd [NOME_DO_REPOSITORIO]
    ```

2.  **Configurar o Ambiente:**
    *   Recomendamos o uso de um ambiente virtual.
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```
    *   Instale as dependências:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Configurar Variáveis de Ambiente:**
    *   Copie o template:
        ```bash
        cp env_template .env
        ```
    *   Edite o arquivo `.env` e adicione seu e-mail. Isso é usado para acessar a "polite pool" da API OpenAlex.

4.  **Colocar os Dados Iniciais:**
    *   Coloque o arquivo CSV com a lista de pesquisadores em `data/raw/author.csv`.
    *   Coloque o arquivo CSV com a lista de PPGs em `data/raw/ppg.csv`.

5.  **Executar o Pipeline de Dados e Análise:**
    *   **(Execução Manual, Passo a Passo)**
        1.  **Enriquecimento de Dados:**
            ```bash
            python -m src.data_collection.get_author_data
            ```
            Isso irá popular a pasta `data/processed/`.
        2.  **Geração de Embeddings:**
            ```bash
            python -m src.feature_engineering.create_embeddings
            ```
        3.  **Execução das Análises:**
            *   Abra e execute os notebooks na pasta `/notebooks` na ordem numérica. Eles carregarão os dados processados e gerarão as figuras em `/reports/figures/`.

## Análises Realizadas

Descreva brevemente cada uma das questões de pesquisa e como os notebooks as abordam.

*   **RQ 1: Validação do Espaço Semântico e Assinatura de Impacto** (ver `01_...ipynb`)
*   **RQ 2: Análise de Rede, Geografia e Prestígio** (ver `02_...ipynb`)

## Ferramentas e Bibliotecas

Liste as principais ferramentas utilizadas (Python, Pandas, PyTorch, Hugging Face Transformers, OpenAlex API, etc.).
