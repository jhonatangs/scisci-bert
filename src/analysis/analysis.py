import pandas as pd
import numpy as np
import hdbscan
import umap
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
import os
import re

# --- 0. Configuração Inicial ---
warnings.filterwarnings("ignore")

# Garante que o diretório para salvar as imagens exista
os.makedirs("images", exist_ok=True)
print("Diretório 'images/' pronto.")


# --- 1. Carregamento dos Dados (sem alterações) ---
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


# --- 2. Pré-processamento e Limpeza (sem alterações) ---
def preprocess_data(df):
    """Realiza o pré-processamento dos dados para a análise."""
    print("Iniciando pré-processamento dos dados...")
    df = df.dropna(subset=["embedding"])
    try:
        df["embedding"] = df["embedding"].apply(eval).apply(np.array)
    except Exception as e:
        print(
            f"Erro ao converter a coluna 'embedding'. Presumindo que já esteja no formato correto. Erro: {e}"
        )
    df["has_productivity_grant"] = df["productivity_grant"].astype(bool)
    for col in ["works_count", "cited_by_count", "h_index", "i10_index", "ppg_score"]:
        df[col] = df[col].fillna(0).astype(int)
    print("Pré-processamento concluído.")
    print(f"Número de registros válidos: {len(df)}")
    return df


# --- 3. Normalização e Redução de Dimensionalidade (sem alterações) ---
def reduce_and_normalize_embeddings(
    embeddings_list, n_neighbors=15, min_dist=0.1, n_components=2
):
    """Normaliza os embeddings e reduz a dimensionalidade para visualização."""
    print("Normalizando e reduzindo a dimensionalidade dos embeddings...")
    if not embeddings_list:
        print("Erro: A lista de embeddings está vazia.")
        return None, None

    embeddings_array = np.vstack(embeddings_list)
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings_array)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(normalized_embeddings)
    print("Redução de dimensionalidade com UMAP concluída.")
    return embedding_2d, normalized_embeddings


# --- 4. Clusterização com HDBSCAN (sem alterações) ---
def cluster_embeddings(embeddings, min_cluster_size=15, min_samples=5):
    """Aplica o HDBSCAN para clusterizar os embeddings."""
    print("Executando a clusterização com HDBSCAN...")
    hdbscan_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    cluster_labels = hdbscan_clusterer.fit_predict(embeddings)
    print(f"Número de clusters encontrados: {len(np.unique(cluster_labels)) - 1}")
    print(f"Número de outliers: {np.sum(cluster_labels == -1)}")
    return cluster_labels


# --- 5. Análise Hierárquica e Comparativa (NOVA SEÇÃO) ---
def characterize_clusters(df: pd.DataFrame, top_n_concepts=5):
    """
    Analisa a coluna 'author_document' para extrair os temas mais comuns
    de cada cluster e retorna um dicionário de rótulos.
    """
    print("\n--- PASSO 1: Caracterizando Clusters Tematicamente ---")
    cluster_labels = {}

    # Adicionado para garantir que 'author_document' seja uma string
    df["author_document"] = df["author_document"].astype(str)

    for cluster_id in sorted(df["cluster"].unique()):
        if cluster_id == -1:
            cluster_labels[cluster_id] = "Outliers & Interdisciplinares"
            continue

        # Filtra o dataframe para o cluster atual
        cluster_docs = df[df["cluster"] == cluster_id]["author_document"]

        # Concatena todos os documentos do cluster e separa os conceitos
        full_text = " ".join(cluster_docs)
        concepts = re.findall(r"\b\w+\b", full_text.lower())

        if not concepts:
            label = f"Cluster {cluster_id}: (Sem conceitos suficientes)"
        else:
            # Conta os conceitos mais frequentes
            concept_counts = Counter(concepts)
            top_concepts = [
                word for word, count in concept_counts.most_common(top_n_concepts)
            ]
            label = (
                f"Cluster {cluster_id}: "
                + ", ".join(top_concepts).replace("_", " ").title()
            )

        cluster_labels[cluster_id] = label
        print(f"  - {label}")

    return cluster_labels


def run_refined_analysis(df: pd.DataFrame):
    """
    Executa a análise comparativa refinada das métricas, usando os rótulos
    temáticos dos clusters.
    """
    print("\n--- PASSO 2: Análise Comparativa Refinada ---")

    # Mapeia os rótulos para uma nova coluna para usar nos gráficos
    cluster_labels = characterize_clusters(df)
    df["cluster_label"] = df["cluster"].map(cluster_labels)

    # Identifica o mega-cluster (mainstream)
    mega_cluster_id = df["cluster"].value_counts().idxmax()
    mega_cluster_label = cluster_labels[mega_cluster_id]
    print(
        f"\nCluster 'Mainstream' identificado: {mega_cluster_label} (ID: {mega_cluster_id})"
    )

    # Visualização Geral com Rótulos
    fig_clusters = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_label",
        hover_data=["author_name", "institution_acr", "h_index"],
        title="Visualização dos Clusters Temáticos de Pesquisadores (UMAP)",
        labels={"cluster_label": "Cluster Temático"},
    )
    fig_clusters.write_html("images/clusters_tematicos_visualization.html")
    fig_clusters.write_image("images/clusters_tematicos_visualization.png")

    # Análise 1: Bolsas de Produtividade
    print("\nAnalisando a distribuição de Bolsas de Produtividade...")
    grant_counts = (
        df.groupby("cluster_label")["has_productivity_grant"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    grant_counts = grant_counts.rename(
        columns={True: "proportion_with_grant", False: "proportion_without_grant"}
    )
    print(
        grant_counts[["proportion_with_grant"]].sort_values(
            by="proportion_with_grant", ascending=False
        )
    )

    # Análise 2: Métricas de Produtividade (comparando com o mainstream)
    print("\nAnalisando Métricas de Produtividade (Mediana)...")
    metrics = ["works_count", "cited_by_count", "h_index", "i10_index"]
    for metric in metrics:
        median_metric = (
            df.groupby("cluster_label")[metric].median().sort_values(ascending=False)
        )
        print(f"\n--- Mediana de '{metric}' por Cluster ---")
        print(median_metric)

        fig_metric = px.box(
            df,
            x=metric,
            y="cluster_label",
            orientation="h",
            title=f"Distribuição de {metric.replace('_', ' ').title()} por Cluster Temático",
            labels={"cluster_label": "Cluster Temático"},
        )
        fig_metric.update_layout(yaxis={"categoryorder": "total ascending"})
        fig_metric.write_html(f"images/{metric}_by_themed_cluster.html")
        fig_metric.write_image(f"images/{metric}_by_themed_cluster.png")

    # Análise 3: Diversidade em PPGs de Alta Nota
    print("\nAnalisando Diversidade de Clusters em PPGs de Alta Nota (6 ou 7)...")
    df_high_score = df[df["ppg_score"].isin([6, 7])].copy()
    if not df_high_score.empty:
        ppg_diversity = (
            df_high_score.groupby("institution_acr")["cluster"]
            .nunique()
            .sort_values(ascending=False)
        )
        print("\nNúmero de Clusters Únicos por PPG de Alta Nota:")
        print(ppg_diversity)

        # Detalha a diversidade do PPG mais diverso
        top_ppg_acr = ppg_diversity.index[0]
        print(f"\nDetalhando a diversidade do PPG mais diverso: {top_ppg_acr}")
        print(
            df_high_score[df_high_score["institution_acr"] == top_ppg_acr][
                "cluster_label"
            ].value_counts()
        )

    return mega_cluster_id


def run_mega_cluster_reclustering(df: pd.DataFrame, mega_cluster_id: int):
    """
    Isola o mega-cluster e executa um novo pipeline de clusterização
    para descobrir sua subestrutura interna.
    """
    print("\n--- PASSO 3 (Avançado): Re-clusterização do Mega-Cluster ---")
    df_mega = df[df["cluster"] == mega_cluster_id].copy()
    print(
        f"Analisando a subestrutura de {len(df_mega)} pesquisadores do cluster mainstream."
    )

    if len(df_mega) < 30:  # Limite mínimo para uma nova clusterização razoável
        print("Mega-cluster é muito pequeno para uma re-clusterização significativa.")
        return

    # Extrai os embeddings originais do subconjunto
    embeddings_list_mega = df_mega["embedding"].tolist()

    # Executa o pipeline de redução e clusterização novamente
    embedding_2d_mega, normalized_embeddings_mega = reduce_and_normalize_embeddings(
        embeddings_list_mega, n_neighbors=10, min_dist=0.05
    )
    sub_clusters = cluster_embeddings(
        normalized_embeddings_mega, min_cluster_size=10, min_samples=3
    )

    # Adiciona os resultados da sub-clusterização ao dataframe
    df_mega["sub_x"] = embedding_2d_mega[:, 0]
    df_mega["sub_y"] = embedding_2d_mega[:, 1]
    df_mega["sub_cluster"] = sub_clusters

    # --- INÍCIO DA CORREÇÃO ---
    # Cria um DataFrame temporário e limpo SÓ para a caracterização,
    # evitando o conflito de nomes de colunas.
    df_for_characterization = df_mega[["author_document", "sub_cluster"]].copy()
    df_for_characterization.rename(columns={"sub_cluster": "cluster"}, inplace=True)

    # Chama a função com o DataFrame limpo
    sub_cluster_labels = characterize_clusters(df_for_characterization)
    # --- FIM DA CORREÇÃO ---

    df_mega["sub_cluster_label"] = df_mega["sub_cluster"].map(sub_cluster_labels)

    print("\nSub-clusters encontrados dentro do Mainstream:")
    print(df_mega["sub_cluster_label"].value_counts())

    fig_sub_clusters = px.scatter(
        df_mega,
        x="sub_x",
        y="sub_y",
        color="sub_cluster_label",
        hover_data=["author_name", "institution_acr", "h_index"],
        title="Sub-Clusters Dentro do Grupo Mainstream",
        labels={"sub_cluster_label": "Sub-Cluster Temático"},
    )
    fig_sub_clusters.write_html("images/sub_clusters_visualization.html")
    fig_sub_clusters.write_image("images/sub_clusters_visualization.png")


# --- 6. Execução do Pipeline Completo ---
def main():
    """Função principal que orquestra todo o pipeline de análise."""
    file_path = "data/processed/featured_authors.parquet"  # Substitua pelo caminho do seu arquivo

    # Etapa 1: Carregar e pré-processar dados
    df_original = load_data(file_path)
    if df_original is None:
        return
    df_processed = preprocess_data(df_original)

    # Etapa 2: Clusterização Primária
    embeddings_list = df_processed["embedding"].tolist()
    embedding_2d, normalized_embeddings = reduce_and_normalize_embeddings(
        embeddings_list
    )
    cluster_labels = cluster_embeddings(normalized_embeddings)

    # Adicionar resultados da clusterização primária ao DataFrame
    df_clustered = df_processed.copy()
    df_clustered["x"] = embedding_2d[:, 0]
    df_clustered["y"] = embedding_2d[:, 1]
    df_clustered["cluster"] = cluster_labels

    # Etapa 3: Análise Refinada e Identificação do Mega-Cluster
    mega_cluster_id = run_refined_analysis(df_clustered)

    # Etapa 4: Re-clusterização do Mega-Cluster
    run_mega_cluster_reclustering(df_clustered, mega_cluster_id)

    print("\n\n--- ANÁLISE CONCLUÍDA ---")
    print("Resultados e gráficos foram salvos na pasta 'images/'.")
    print("Use os arquivos .html para visualizações interativas.")


if __name__ == "__main__":
    main()
