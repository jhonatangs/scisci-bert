
# Explorando a Rede Científica de Computação no Brasil

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-em%20desenvolvimento-orange)

Este repositório contém o código-fonte, os dados processados e as análises para o projeto de pesquisa que mapeia o cenário da ciência da computação no Brasil. O projeto utiliza dados de fontes como a **CAPES**, a **Plataforma Lattes** e a **OpenAlex**, combinados com técnicas de Processamento de Linguagem Natural (**BERT**) e Ciência de Redes para descobrir padrões sobre impacto, colaboração e a estrutura da comunidade científica nacional.

## 📖 Visão Geral do Projeto

O objetivo central deste trabalho é transformar o perfil de publicações de cada pesquisador em uma representação vetorial densa (um *embedding semântico*). Este "mapa semântico" nos permite ir além das métricas tradicionais e investigar questões como:

*   Quais são as verdadeiras comunidades de pesquisa que emergem dos tópicos estudados?
*   A colaboração científica é guiada pela proximidade temática, geográfica ou institucional?
*   Pesquisadores de alto impacto ocupam posições específicas neste mapa (e.g., no centro de um cluster ou na intersecção de vários)?
*   Qual o papel das instituições de excelência na formação e conexão da rede científica nacional?

## ✨ Principais Características

*   **Fonte de Dados Integrada:** Combina dados institucionais (CAPES), curriculares (Lattes) e bibliográficos globais (OpenAlex).
*   **Representação Semântica:** Utiliza o modelo **SciBERT**, treinado em textos científicos, para gerar embeddings que capturam o significado dos trabalhos de um pesquisador.
*   **Pipeline Reprodutível:** Todo o processo, desde a coleta de dados até a geração de gráficos, é automatizado e documentado para garantir total reprodutibilidade.
*   **Análise Multifacetada:** Investiga a estrutura da ciência brasileira sob as óticas de clustering, análise de redes e correlação com métricas de impacto.

## 📂 Estrutura do Repositório

O projeto é organizado de forma modular para separar as responsabilidades e facilitar a manutenção.

```bash
/explorando-rede-cientifica-br/
|
|-- data/
| |-- raw/ # (Intocável) Dados brutos, como o CSV inicial de pesquisadores.
| |-- processed/ # Dados limpos, enriquecidos e prontos para análise.
|
|-- notebooks/ # Notebooks Jupyter para análise exploratória e visualização dos resultados.
| |-- 01_Semantic_Space_Validation.ipynb
| |-- 02_Network_and_Collaboration_Analysis.ipynb
| |-- ...
|
|-- src/ # Código-fonte principal do projeto.
| |-- data_collection/ # Módulos para coletar e enriquecer dados da API OpenAlex.
| |-- feature_engineering/# Módulos para gerar embeddings e outras features.
| |-- analysis/ # Funções para clustering, análise de redes, estatísticas.
| |-- visualization/ # Funções de plotagem reutilizáveis.
| |-- utils/ # Funções auxiliares, configurações, logging.
|
|-- reports/
| |-- figures/ # Gráficos e visualizações gerados para o relatório final.
|
|-- .env_template # Template para variáveis de ambiente (e.g., e-mail da API).
|-- requirements.txt # Dependências do projeto para pip.
|-- README.md # Esta documentação.
```
Generated code
## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar o ambiente e reproduzir todas as análises.

### 1. Pré-requisitos

*   [Git](https://git-scm.com/) instalado.
*   [Python](https://www.python.org/downloads/) (versão 3.9 ou superior).
*   Recomendado: `venv` para criar um ambiente virtual isolado.

### 2. Clonar e Configurar o Ambiente

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/explorando-rede-cientifica-br.git
cd explorando-rede-cientifica-br

# 2. Crie e ative um ambiente virtual
python -m venv venv
# No Linux/macOS:
source venv/bin/activate
# No Windows:
.\venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
3. Configurar Variáveis de Ambiente

Para usar a API da OpenAlex de forma "polida" (obtendo acesso mais rápido e estável), é recomendado fornecer um endereço de e-mail.

Generated bash
# Copie o template para um arquivo .env
cp .env_template .env
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Agora, abra o arquivo .env com um editor de texto e insira seu e-mail:

Generated code
OPENALEX_EMAIL="seu-email@exemplo.com"
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
4. Preparar os Dados Iniciais

Coloque o arquivo CSV inicial, contendo a lista de pesquisadores a serem analisados, dentro da pasta data/raw/. O arquivo deve ser nomeado como researchers.csv.

5. Executar o Pipeline Completo

O pipeline executa três etapas principais em sequência:

Coleta de Dados: Busca os dados de cada pesquisador na OpenAlex.

Engenharia de Features: Gera os embeddings semânticos com base nos textos coletados.

Análise e Visualização: Executa as análises e gera os resultados.

Para executar todo o processo, utilize os scripts principais de cada módulo na ordem correta:

Generated bash
# Passo 1: Coletar dados da OpenAlex e enriquecer o dataset
# (Isso pode levar um tempo considerável dependendo do número de pesquisadores)
python -m src.data_collection.main

# Passo 2: Gerar os embeddings para cada pesquisador
# (Requer uma GPU com CUDA para melhor desempenho)
python -m src.feature_engineering.main

# Passo 3: Executar as análises e gerar os resultados
# Abra e execute os notebooks na pasta /notebooks em ordem numérica (01, 02, ...).
# Os notebooks carregarão os dados de /data/processed/ e salvarão os gráficos em /reports/figures/.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
🔬 Questões de Pesquisa e Análises

As análises estão organizadas em notebooks, cada um focado em responder a um conjunto de questões de pesquisa:

notebooks/01_Semantic_Space_Validation.ipynb

RQ 1.1: Valida se os clusters semânticos correspondem às áreas de pesquisa formais (CAPES/SBC).

RQ 1.2: Investiga se pesquisadores de alto impacto (Bolsistas PQ, alto índice-h) ocupam posições distintas no espaço semântico.

notebooks/02_Network_and_Collaboration_Analysis.ipynb

RQ 2.1: Analisa a correlação entre proximidade semântica e colaboração (coautoria).

RQ 2.2: Explora a influência da "linhagem acadêmica" (instituição de doutorado) na organização do mapa temático.

RQ 2.3: Examina o papel dos programas de excelência (nota 6/7) como "hubs" ou "ilhas" de conhecimento.

🛠️ Tecnologia e Ferramentas

Linguagem: Python 3.9+

Análise de Dados: Pandas, NumPy, Scikit-learn

API e Redes: OpenAlex API, Pyalex, NetworkX

NLP & Embeddings: PyTorch, Hugging Face Transformers (SciBERT)

Visualização: Matplotlib, Seaborn, Plotly

Redução de Dimensionalidade: UMAP

📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.

✍️ Citação

Se você utilizar este código ou os resultados em sua pesquisa, por favor, cite este trabalho da seguinte forma:

SOUZA, J. G.; LUZ, E. J. S.; FREITAS, V. L. S. Explorando a Rede Científica de Computação no Brasil: Abordagem Integrada com BERT, Análise de Agrupamento e OpenAlex. Monografia (Bacharelado em Ciência da Computação) - Universidade Federal de Ouro Preto, Ouro Preto, 2025.

Generated code
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
