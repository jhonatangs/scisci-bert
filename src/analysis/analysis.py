import pandas as pd
import numpy as np
import hdbscan
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import warnings
import logging

warnings.filterwarnings("ignore")

# Configuração de logging
# logging.basicConfig(
#    level=print, format="%(asctime)s - %(levelname)s - %(message)s"
# )


# --- 1. Carregamento dos Dados ---
def load_data(file_path):
    """Carrega os dados de um arquivo parquet."""
    print(f"Carregando dados de {file_path}...")
    try:
        df = pd.read_parquet(file_path)
        print("Dados carregados com sucesso!")
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return None


# --- 2. Pré-processamento e Limpeza ---
def preprocess_data(df):
    """Realiza o pré-processamento dos dados para a análise."""
    print("Iniciando pré-processamento dos dados...")

    # Remover linhas com valores nulos nos embeddings (se houver)
    df = df.dropna(subset=["embedding"])

    try:
        df["embedding"] = df["embedding"].apply(eval).apply(np.array)
    except Exception as e:
        print(
            f"Erro ao converter a coluna 'embedding'. Presumindo que já esteja no formato correto. Erro: {e}"
        )

    # Criar uma coluna binária para a bolsa de produtividade
    df["has_productivity_grant"] = df["productivity_grant"].astype(bool)

    # Tratar valores nulos para as métricas de produtividade
    for col in ["works_count", "cited_by_count", "h_index", "i10_index", "ppg_score"]:
        df[col] = df[col].fillna(0).astype(int)

    print("Pré-processamento concluído.")
    print(f"Número de registros válidos: {len(df)}")

    return df


# --- 3. Normalização e Redução de Dimensionalidade ---
def reduce_and_normalize_embeddings(df, n_neighbors=15, min_dist=0.1, n_components=2):
    """Normaliza os embeddings e reduz a dimensionalidade para visualização."""
    print("Normalizando e reduzindo a dimensionalidade dos embeddings...")
    embeddings_list = df["embedding"].tolist()
    # Verificar se a lista de embeddings não está vazia
    if not embeddings_list:
        print("Erro: A lista de embeddings está vazia.")
        return df, None

    embeddings_array = np.vstack(embeddings_list)

    # Normalização
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings_array)

    # Redução de dimensionalidade com UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
    )

    # Adicione um bloco try-except para capturar possíveis erros do UMAP
    try:
        embedding_2d = reducer.fit_transform(normalized_embeddings)
    except Exception as e:
        print(f"Erro ao executar UMAP: {e}")
        return df, normalized_embeddings

    df["x"] = embedding_2d[:, 0]
    df["y"] = embedding_2d[:, 1]

    print("Redução de dimensionalidade com UMAP concluída.")
    print(f"Valores de X:\n {df['x'].head()}")
    print(f"Valores de Y:\n {df['y'].head()}")

    return df, normalized_embeddings


# --- 4. Clusterização com HDBSCAN ---
def cluster_embeddings(df, embeddings, min_cluster_size=15, min_samples=5):
    """Aplica o HDBSCAN para clusterizar os embeddings."""
    print("Executando a clusterização com HDBSCAN...")
    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    cluster_labels = hdbscan_clusterer.fit_predict(embeddings)

    df["cluster"] = cluster_labels
    df["cluster"] = df["cluster"].astype("category")

    # O HDBSCAN atribui -1 aos outliers, o que é útil
    print(f"Número de clusters encontrados: {len(df['cluster'].unique()) - 1}")
    print(f"Número de outliers: {sum(df['cluster'] == -1)}")

    return df


# --- 5. Análise e Visualização para Responder às Perguntas de Pesquisa ---
def analyze_and_visualize(df):
    """Gera visualizações e análises para responder às perguntas de pesquisa."""
    print("Iniciando a análise e visualização dos resultados...")

    print("Estatísticas das colunas X e Y:")
    print(df[["x", "y"]].describe())
    print("\nContagem de valores na coluna cluster:")
    print(df["cluster"].value_counts())

    # Visualização geral dos clusters
    fig_clusters = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        hover_data=["author_name", "gp_name", "name_ppg", "has_productivity_grant"],
        title="Visualização dos Clusters de Pesquisadores (UMAP)",
        labels={
            "cluster": "Cluster ID",
            "x": "Dimensão 1 (UMAP)",
            "y": "Dimensão 2 (UMAP)",
        },
    )
    fig_clusters.write_image("images/clusters_visualization.png")
    fig_clusters.write_html("images/clusters_visualization.html")

    # --- Pergunta 1: Bolsas de Produtividade CNPq ---
    print("\n--- Análise da Relação entre Bolsas de Produtividade e Clusters ---")

    # Comparação de distribuição de clusters entre grupos
    df["cluster_str"] = df["cluster"].astype(str)
    fig_grants = px.histogram(
        df,
        x="cluster_str",
        color="has_productivity_grant",
        barmode="group",
        title="Distribuição de Pesquisadores com/sem Bolsa de Produtividade por Cluster",
        labels={"cluster_str": "ID do Cluster", "count": "Número de Pesquisadores"},
    )
    fig_grants.write_image("images/grants_distribution.png")
    fig_grants.write_html("images/grants_distribution.html")

    # Análise quantitativa
    grant_counts = (
        df.groupby(["cluster", "has_productivity_grant"]).size().unstack(fill_value=0)
    )
    grant_counts["proportion_with_grant"] = grant_counts[True] / (
        grant_counts[True] + grant_counts[False]
    )
    print(grant_counts.sort_values(by="proportion_with_grant", ascending=False))

    print(
        "Conclusão: Clusters com alta proporção de bolsas podem indicar áreas temáticas priorizadas ou mais estabelecidas."
    )

    # --- Pergunta 2: Relação entre Especialização Temática e Produtividade ---
    print("\n--- Análise da Relação entre Clusters e Métricas de Produtividade ---")

    for metric in ["works_count", "cited_by_count", "h_index", "i10_index"]:
        fig_metric = px.box(
            df,
            x="cluster",
            y=metric,
            title=f"Distribuição de {metric} por Cluster",
            hover_data=["author_name"],
        )
        fig_metric.write_image(f"images/{metric}_by_cluster.png")
        fig_metric.write_html(f"images/{metric}_by_cluster.html")

    print(
        "Conclusão: Observar as distribuições (mediana, quartis) de métricas por cluster ajuda a identificar perfis de alta performance."
    )

    # --- Pergunta 3V2: Homogeneidade vs. Diversidade em PPGs de Alta Nota ---
    print("\n--- Análise de Diversidade de Clusters em PPGs de Alta Nota (6 ou 7) ---")

    df_high_score = df[df["ppg_score"].isin([6, 7])].copy()
    if not df_high_score.empty:
        # Calcular a diversidade (entropia ou número de clusters) para cada PPG de alta nota
        ppg_diversity = df_high_score.groupby("institution_acr")["cluster"].nunique()
        print("\nNúmero de Clusters por PPG de Alta Nota (indicador de diversidade):")
        print(ppg_diversity.sort_values(ascending=False))

        # Visualizar a distribuição de clusters dentro de um PPG específico
        top_ppg = ppg_diversity.idxmax()
        if top_ppg:
            df_top_ppg = df_high_score[df_high_score["institution_acr"] == top_ppg]
            fig_ppg_clusters = px.scatter(
                df_top_ppg,
                x="x",
                y="y",
                color="cluster",
                hover_data=["author_name"],
                title=f"Clusters de Pesquisadores no PPG de Alta Nota: {top_ppg}",
            )
            fig_ppg_clusters.write_image(f"images/{top_ppg}_clusters.png")
            fig_ppg_clusters.write_html(f"images/{top_ppg}_clusters.html")

    print(
        "Conclusão: Se um PPG com alta nota tem muitos clusters, sugere diversidade. Poucos clusters indicam homogeneidade."
    )


# --- 6. Execução do Pipeline Completo ---
def main():
    file_path = "data/processed/featured_authors.parquet"  # Substitua pelo caminho do seu arquivo

    # Executar o pipeline sequencialmente
    df = load_data(file_path)
    if df is not None:
        df = preprocess_data(df)
        df, normalized_embeddings = reduce_and_normalize_embeddings(df)
        df_clustered = cluster_embeddings(df, normalized_embeddings)
        analyze_and_visualize(df_clustered)


if __name__ == "__main__":
    main()
