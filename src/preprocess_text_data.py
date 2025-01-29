import os
import gdown
import pandas as pd
import re

def load_dataset(csv_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset successfully loaded. Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        return df
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def remove_missing_values(df, columns):
    """Remove rows with missing values in specified columns."""
    initial_shape = df.shape
    df_clean = df.dropna(subset=columns)
    print(f"Rows removed: {initial_shape[0] - df_clean.shape[0]}")
    return df_clean

def clean_text(text):
    """Clean text by converting to lowercase, removing punctuation, and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_preprocessed_data(df, save_path):
    """Save the processed DataFrame to a CSV file."""
    try:
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    # ------------------------------
    # 1) Create the output directory if it doesn't exist
    # ------------------------------
    output_dir = "/content/arxiv-GCN-TopicClustering/data/processed"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    # ------------------------------
    # 2) Load the datasets from the specified paths
    # ------------------------------
    edges_df_path = '/content/arxiv-GCN-TopicClustering/data/processed/edges_sub.csv'
    papers_df_path = '/content/arxiv-GCN-TopicClustering/data/processed/papers_sub.csv'
    labels_df_path = '/content/arxiv-GCN-TopicClustering/data/processed/labels_sub.csv'

    edges_df = load_dataset(edges_df_path)
    papers_df = load_dataset(papers_df_path)
    labels_df = load_dataset(labels_df_path)

    if edges_df is None or papers_df is None or labels_df is None:
        return

    # ------------------------------
    # 3) Extract valid node IDs from papers
    # ------------------------------
    valid_nodes = set(papers_df['node_id'].astype(str))
    print(f"Valid nodes: {len(valid_nodes)}")

    # ------------------------------
    # 4) Process the edges data
    # ------------------------------
    edges_df['src'] = edges_df['src'].astype(str)
    edges_df['dst'] = edges_df['dst'].astype(str)

    edges_sub_df = edges_df[
        (edges_df['src'].isin(valid_nodes)) & 
        (edges_df['dst'].isin(valid_nodes))
    ].reset_index(drop=True)

    print(f"Filtered edges: {edges_sub_df.shape[0]}")
    edges_preprocessed_csv_path = os.path.join(output_dir, "edges_preprocessed.csv")
    save_preprocessed_data(edges_sub_df, edges_preprocessed_csv_path)

    # ------------------------------
    # 5) Process the labels data
    # ------------------------------
    labels_df['node_id'] = labels_df['node_id'].astype(str)
    labels_sub_df = labels_df[labels_df['node_id'].isin(valid_nodes)].reset_index(drop=True)

    print(f"Filtered labels: {labels_sub_df.shape[0]}")
    labels_preprocessed_csv_path = os.path.join(output_dir, "labels_preprocessed.csv")
    save_preprocessed_data(labels_sub_df, labels_preprocessed_csv_path)

    # ------------------------------
    # 6) Process the papers data
    # ------------------------------
    text_columns = ['title', 'abstract']
    papers_clean_df = remove_missing_values(papers_df, text_columns)

    # Clean text in 'title' and 'abstract'
    for col in text_columns:
        print(f"Cleaning column: {col}...")
        papers_clean_df[col] = papers_clean_df[col].apply(clean_text)
    print("Text cleaning done.")

    papers_preprocessed_csv_path = os.path.join(output_dir, "papers_preprocessed.csv")
    save_preprocessed_data(papers_clean_df, papers_preprocessed_csv_path)

    print("Processing complete!")

if __name__ == "__main__":
    main()
