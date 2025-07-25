import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def prepare_author_document(author_data: list, strategy: str = 'x_concepts') -> str:
    """
    Cria um "documento" único que representa o autor a partir de seus conceitos.

    Args:
        author_data (list): Uma lista de dicionários, onde cada dicionário representa um conceito.
        strategy (str, optional): A estratégia para criar o documento. Defaults to 'x_concepts'.

    Returns:
        str: Uma string com os nomes dos conceitos concatenados.
    """
    if not isinstance(author_data, list):
        return ""

    if strategy == 'x_concepts':
        # Concatena os nomes dos conceitos, ponderando pela repetição baseada no score
        concepts = []
        for concept in author_data:
            if 'display_name' in concept and 'score' in concept:
                # Repete o nome do conceito com base no score (arredondado para o inteiro mais próximo)
                repeat_count = int(round(concept['score'] * 10))
                concepts.extend([concept['display_name']] * repeat_count)
        return " ".join(concepts)

    return ""

def generate_bert_embeddings(documents: list, model_name: str = 'allenai/scibert_scivocab_uncased', device: str = 'cpu') -> np.ndarray:
    """
    Gera embeddings BERT para uma lista de documentos.

    Args:
        documents (list): Uma lista de strings (documentos).
        model_name (str, optional): O nome do modelo da Hugging Face. Defaults to 'allenai/scibert_scivocab_uncased'.
        device (str, optional): O dispositivo para rodar o modelo ('cuda' ou 'cpu'). Defaults to 'cpu'.

    Returns:
        np.ndarray: Uma matriz NumPy com os embeddings.
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    embeddings = []
    for doc in tqdm(documents, desc="Gerando Embeddings"):
        inputs = tokenizer(doc, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Usa o embedding do token [CLS]
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding)

    return np.vstack(embeddings)

def build_feature_set(processed_data_path: str, output_path: str):
    """
    Carrega os dados processados e cria o conjunto de features.

    Args:
        processed_data_path (str): O caminho para o arquivo parquet com os dados enriquecidos.
        output_path (str): O caminho para salvar o DataFrame final com as features.
    """
    try:
        enriched_df = pd.read_parquet(processed_data_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo de dados processados não encontrado em '{processed_data_path}'")
        return

    documents = []
    for _, row in tqdm(enriched_df.iterrows(), total=enriched_df.shape[0], desc="Preparando documentos"):
        doc = prepare_author_document(row.get('x_concepts'))
        documents.append(doc)

    enriched_df['author_document'] = documents

    # Gera os embeddings
    embeddings = generate_bert_embeddings(documents)
    enriched_df['embedding'] = list(embeddings)

    # Salva o DataFrame com a nova coluna de embeddings
    enriched_df.to_parquet(output_path)
    print(f"Conjunto de features salvo em: {output_path}")

if __name__ == '__main__':
    processed_parquet = 'data/processed/enriched_authors.parquet'
    features_parquet = 'data/processed/featured_authors.parquet'

    # Para testar, vamos criar um dataframe de exemplo, já que a coleta de dados pode não ter sido executada
    if not os.path.exists(processed_parquet):
        print("Criando dados de amostra para teste...")
        data = {
            'id': ['A1', 'A2'],
            'x_concepts': [
                [{'display_name': 'Machine Learning', 'score': 0.8}, {'display_name': 'Artificial Intelligence', 'score': 0.7}],
                [{'display_name': 'Computer Networks', 'score': 0.9}, {'display_name': 'Security', 'score': 0.6}]
            ]
        }
        sample_df = pd.DataFrame(data)
        os.makedirs('data/processed', exist_ok=True)
        sample_df.to_parquet(processed_parquet)

    build_feature_set(processed_parquet, features_parquet)
