from get_author_data import enrich_researcher_data

if __name__ == "__main__":
    authors_csv_path = 'data/raw/author.csv'
    ppgs_csv_path = 'data/raw/ppg.csv'
    output_parquet_path = 'data/processed/enriched_authors.parquet'

    enrich_researcher_data(authors_csv_path, ppgs_csv_path, output_parquet_path)
