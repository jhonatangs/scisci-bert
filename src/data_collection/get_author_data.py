import pandas as pd
import pyalex
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configura o e-mail para a API da OpenAlex
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")
if OPENALEX_EMAIL:
    pyalex.config.email = OPENALEX_EMAIL

def load_initial_researchers(csv_path_authors: str, csv_path_ppgs: str) -> pd.DataFrame:
    """
    Carrega os CSVs iniciais de autores e PPGs em DataFrames Pandas e os une.

    Args:
        csv_path_authors (str): O caminho para o arquivo CSV de autores.
        csv_path_ppgs (str): O caminho para o arquivo CSV de PPGs.

    Returns:
        pd.DataFrame: Um DataFrame contendo os dados combinados dos pesquisadores.
    """
    try:
        authors_df = pd.read_csv(csv_path_authors)
        ppgs_df = pd.read_csv(csv_path_ppgs)

        # Renomeia colunas para a junção
        ppgs_df.rename(columns={'code_ppg': 'gp_code'}, inplace=True)

        # Une os dois dataframes
        researchers_df = pd.merge(authors_df, ppgs_df, on='gp_code', how='left')

        print(f"Carregados {len(researchers_df)} pesquisadores.")
        return researchers_df
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}")
        return pd.DataFrame()

def find_openalex_author(name: str, affiliation: str = None) -> dict | None:
    """
    Consulta a API OpenAlex para encontrar o perfil de um pesquisador.

    Args:
        name (str): O nome completo do pesquisador.
        affiliation (str, optional): A afiliação institucional do pesquisador.

    Returns:
        dict | None: Um dicionário com os dados do autor da OpenAlex ou None se não for encontrado.
    """
    try:
        # Tenta uma busca exata pelo nome
        authors = pyalex.Authors().search_filter(display_name=name).get()

        if not authors:
            return None

        # Estratégia de desambiguação
        if len(authors) > 1 and affiliation:
            # Filtra por afiliação, se disponível
            for author in authors:
                if author.get('last_known_institution') and affiliation.lower() in author.get('last_known_institution', {}).get('display_name', '').lower():
                    return author

        # Heurístico de desempate: retorna o autor com mais trabalhos
        return max(authors, key=lambda x: x.get('works_count', 0))

    except Exception as e:
        print(f"Erro ao buscar autor '{name}': {e}")
        return None

def fetch_author_details(author_id: str) -> dict | None:
    """
    Busca metadados detalhados de um autor na OpenAlex.

    Args:
        author_id (str): O ID do autor na OpenAlex (e.g., A12345678).

    Returns:
        dict | None: Um dicionário com os detalhes do autor ou None se ocorrer um erro.
    """
    try:
        author = pyalex.Authors()[author_id]
        details = {
            'id': author.get('id'),
            'display_name': author.get('display_name'),
            'orcid': author.get('orcid'),
            'works_count': author.get('works_count'),
            'cited_by_count': author.get('cited_by_count'),
            'h_index': author.get('summary_stats', {}).get('h_index'),
            'i10_index': author.get('summary_stats', {}).get('i10_index'),
            'last_known_institution': author.get('last_known_institution'),
            'x_concepts': author.get('x_concepts'),
        }
        return details
    except Exception as e:
        print(f"Erro ao buscar detalhes do autor '{author_id}': {e}")
        return None

def enrich_researcher_data(input_csv_authors: str, input_csv_ppgs: str, output_path: str):
    """
    Orquestra o processo de enriquecimento de dados dos pesquisadores.

    Args:
        input_csv_authors (str): O caminho para o arquivo CSV de autores.
        input_csv_ppgs (str): O caminho para o arquivo CSV de PPGs.
        output_path (str): O caminho para salvar o DataFrame enriquecido em formato Parquet.
    """
    researchers_df = load_initial_researchers(input_csv_authors, input_csv_ppgs)
    if researchers_df.empty:
        return

    enriched_data = []
    for _, row in tqdm(researchers_df.iterrows(), total=researchers_df.shape[0], desc="Enriquecendo dados"):
        author_name = row['author_name']
        affiliation_name = row.get('institution_name_acronym')

        openalex_author = find_openalex_author(author_name, affiliation_name)

        if openalex_author and openalex_author.get('id'):
            author_id = openalex_author.get('id')
            details = fetch_author_details(author_id)
            if details:
                row_data = row.to_dict()
                row_data.update(details)
                enriched_data.append(row_data)

        time.sleep(0.1)  # Respeita a política de uso da API

    enriched_df = pd.DataFrame(enriched_data)

    if not enriched_df.empty:
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        enriched_df.to_parquet(output_path)
        print(f"Dados enriquecidos salvos em: {output_path}")
    else:
        print("Nenhum dado enriquecido para salvar.")


if __name__ == '__main__':
    authors_csv = 'data/raw/author.csv'
    ppgs_csv = 'data/raw/ppg.csv'
    output_parquet = 'data/processed/enriched_authors.parquet'

    enrich_researcher_data(authors_csv, ppgs_csv, output_parquet)
