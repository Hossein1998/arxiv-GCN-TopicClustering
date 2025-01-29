import pandas as pd
import re
import networkx as nx
import random
import gdown
import os

def load_dataset(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset successfully loaded. Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
        return df
    except FileNotFoundError:
        print(f"File not found at path: {csv_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

def remove_missing_values(df, columns):
    """Remove rows with missing values in specified columns."""
    initial_shape = df.shape
    df_clean = df.dropna(subset=columns)
    final_shape = df_clean.shape
    dropped_rows = initial_shape[0] - final_shape[0]
    print(f"Number of rows removed due to missing values: {dropped_rows}")
    return df_clean

def clean_text(text):
    """Clean text by lowercasing, removing punctuation, and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_preprocessed_data(df, save_path):
    """Save preprocessed DataFrame to CSV."""
    try:
        df.to_csv(save_path, index=False)
        print(f"Preprocessed DataFrame saved to {save_path}")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive."""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        print(f"File downloaded successfully and saved to {destination}")
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")

def main():
    # ------------------------------
    # 0) Download paper_info.csv from Google Drive
    # ------------------------------
    FILE_ID = "1qhsP1zqjQ2IAwKKzdkpcJ4T1y-2SGrUJ"
    DESTINATION = "data/raw/paper_info.csv"

    os.makedirs(os.path.dirname(DESTINATION), exist_ok=True)

    if not os.path.exists(DESTINATION):
        print("Downloading paper_info.csv from Google Drive...")
        download_file_from_google_drive(FILE_ID, DESTINATION)
    else:
        print(f"paper_info.csv already exists at {DESTINATION}")

    # ------------------------------
    # 1) Load Datasets
    # ------------------------------
    papers_csv_path = DESTINATION
    papers_df = load_dataset(papers_csv_path)
    if papers_df is None:
        return

    # ------------------------------
    # 2) Remove Missing Values
    # ------------------------------
    text_columns = ['title', 'abstract']
    papers_clean_df = remove_missing_values(papers_df, text_columns)

    # ------------------------------
    # 3) Clean Text Columns
    # ------------------------------
    for col in text_columns:
        print(f"Cleaning text in column '{col}'...")
        papers_clean_df[col] = papers_clean_df[col].apply(clean_text)
    print("Text cleaning completed.")

    # ------------------------------
    # 4) Ensure Processed Directory Exists
    # ------------------------------
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # ------------------------------
    # 5) Save Preprocessed Paper Data
    # ------------------------------
    paper_preprocessed_path = f"{processed_dir}/paper_info_preprocessed.csv"
    save_preprocessed_data(papers_clean_df, paper_preprocessed_path)

    # ------------------------------
    # 6) Filter Edges
    # ------------------------------
    valid_nodes = set(papers_clean_df['node_id'].astype(str))
    print(f"Number of valid nodes after cleaning: {len(valid_nodes)}")

    edges_csv_path = "data/raw/edge_list.csv"
    edges_preprocessed_path = f"{processed_dir}/edges_preprocessed.csv"

    edges_df = load_dataset(edges_csv_path)
    if edges_df is None:
        return

    edges_df['src'] = edges_df['src'].astype(str)
    edges_df['dst'] = edges_df['dst'].astype(str)

    edges_sub_df = edges_df[
        (edges_df['src'].isin(valid_nodes)) & (edges_df['dst'].isin(valid_nodes))
    ].reset_index(drop=True)

    print(f"Filtered edges count: {edges_sub_df.shape[0]}")
    save_preprocessed_data(edges_sub_df, edges_preprocessed_path)

    # ------------------------------
    # 7) Filter Labels
    # ------------------------------
    labels_csv_path = "data/raw/labels.csv"
    labels_preprocessed_path = f"{processed_dir}/labels_preprocessed.csv"

    labels_df = load_dataset(labels_csv_path)
    if labels_df is None:
        return

    labels_df['node_id'] = labels_df['node_id'].astype(str)
    labels_sub_df = labels_df[labels_df['node_id'].isin(valid_nodes)].reset_index(drop=True)

    print(f"Filtered labels count: {labels_sub_df.shape[0]}")
    save_preprocessed_data(labels_sub_df, labels_preprocessed_path)

    print("Data preprocessing completed successfully!")

    # ------------------------------
    # 8) Build Graph & Sample Largest Component
    # ------------------------------
    G = nx.from_pandas_edgelist(
        edges_sub_df,
        source='src',
        target='dst',
        create_using=nx.Graph()
    )

    components = nx.connected_components(G)
    largest_component = max(components, key=len)
    desired_size = 57471

    if len(largest_component) > desired_size:
        sampled_nodes = random.sample(largest_component, desired_size)
        print(f"Randomly sampled {desired_size} nodes.")
    else:
        sampled_nodes = list(largest_component)
        print(f"Using all {len(largest_component)} nodes from the largest component.")

    sampled_nodes_set = set(sampled_nodes)

    # ------------------------------
    # 9) Filter Subset Data
    # ------------------------------
    edges_sub_df = edges_sub_df[
        (edges_sub_df['src'].isin(sampled_nodes_set)) &
        (edges_sub_df['dst'].isin(sampled_nodes_set))
    ].reset_index(drop=True)

    papers_sub_df = papers_clean_df[papers_clean_df['node_id'].isin(sampled_nodes_set)].reset_index(drop=True)
    labels_sub_df = labels_sub_df[labels_sub_df['node_id'].isin(sampled_nodes_set)].reset_index(drop=True)

    # ------------------------------
    # 10) Verify & Save Subset Data
    # ------------------------------
    edges_sub_df.to_csv(f"{processed_dir}/edges_sub.csv", index=False)
    papers_sub_df.to_csv(f"{processed_dir}/papers_sub.csv", index=False)
    labels_sub_df.to_csv(f"{processed_dir}/labels_sub.csv", index=False)

    print(f"Saved subset data to {processed_dir}")
    print(f"Selected {len(sampled_nodes)} nodes.")
    print(f"Edges shape: {edges_sub_df.shape}")
    print(f"Papers shape: {papers_sub_df.shape}")
    print(f"Labels shape: {labels_sub_df.shape}")

if __name__ == "__main__":
    main()
