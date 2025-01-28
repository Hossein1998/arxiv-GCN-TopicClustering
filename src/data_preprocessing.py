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
    """
    Remove rows with missing values in specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to check for missing values.

    Returns:
        pd.DataFrame: DataFrame after removing rows with missing values.
    """
    initial_shape = df.shape
    df_clean = df.dropna(subset=columns)
    final_shape = df_clean.shape
    dropped_rows = initial_shape[0] - final_shape[0]  
    return df_clean

def clean_text(text):
    """
    Clean the input text by converting to lowercase, removing punctuation and non-alphanumeric characters,
    and eliminating extra whitespace.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def save_preprocessed_data(df, save_path):
    """
    Save the preprocessed DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        save_path (str): Path to save the CSV file.
    """
    try:
        df.to_csv(save_path, index=False)
        print(f"Preprocessed DataFrame saved to {save_path}")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")

def download_file_from_google_drive(file_id, destination):
    """
    Downloads a file from Google Drive.

    Args:
        file_id (str): The unique identifier for the file on Google Drive.
        destination (str): The path where the downloaded file will be saved.
    """
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
    # Load paper_info.csv with columns ['node_id', 'title', 'abstract']
    papers_csv_path = DESTINATION  # Path updated after download
    papers_preprocessed_csv_path = "data/processed/paper_info_preprocessed.csv"

    papers_df = load_dataset(papers_csv_path)
    if papers_df is None:
        return

    # ------------------------------
    # 2) Remove Rows with Missing Values in 'title' and 'abstract'
    # ------------------------------
    text_columns = ['title', 'abstract']
    papers_clean_df = remove_missing_values(papers_df, text_columns)

    # ------------------------------
    # 3) Clean Text in 'title' and 'abstract' Columns
    # ------------------------------
    for col in text_columns:
        print(f"Cleaning text in column '{col}'...")
        papers_clean_df[col] = papers_clean_df[col].apply(clean_text)
    print("Text cleaning completed.")

    # ------------------------------
    # 4) Save the Preprocessed Paper Information
    # ------------------------------
    save_preprocessed_data(papers_clean_df, papers_preprocessed_csv_path)

    # ------------------------------
    # 5) Identify Valid Nodes After Cleaning
    # ------------------------------
    valid_nodes = set(papers_clean_df['node_id'].astype(str))
    print(f"Number of valid nodes after cleaning: {len(valid_nodes)}")

    # ------------------------------
    # 6) Load and Filter edge_list.csv Based on Valid Nodes
    # ------------------------------
    edges_csv_path = "data/raw/edge_list.csv"  # Update this path as needed
    edges_preprocessed_csv_path = "data/processed/edges_preprocessed.csv"

    edges_df = load_dataset(edges_csv_path)
    if edges_df is None:
        return

    # Ensure consistent data types
    edges_df['src'] = edges_df['src'].astype(str)
    edges_df['dst'] = edges_df['dst'].astype(str)

    # Filter edges where both src and dst are in valid_nodes
    edges_sub_df = edges_df[
        (edges_df['src'].isin(valid_nodes)) &
        (edges_df['dst'].isin(valid_nodes))
    ].reset_index(drop=True)

    print(f"Filtered edges count: {edges_sub_df.shape[0]}")

    # Save the filtered edges
    save_preprocessed_data(edges_sub_df, edges_preprocessed_csv_path)

    # ------------------------------
    # 7) Load and Filter labels.csv Based on Valid Nodes
    # ------------------------------
    labels_csv_path = "data/raw/labels.csv"  # Update this path as needed
    labels_preprocessed_csv_path = "data/processed/labels_preprocessed.csv"

    labels_df = load_dataset(labels_csv_path)
    if labels_df is None:
        return

    # Ensure 'node_id' is of type string
    labels_df['node_id'] = labels_df['node_id'].astype(str)

    # Filter labels where node_id is in valid_nodes
    labels_sub_df = labels_df[labels_df['node_id'].isin(valid_nodes)].reset_index(drop=True)

    print(f"Filtered labels count: {labels_sub_df.shape[0]}")

    # Save the filtered labels
    save_preprocessed_data(labels_sub_df, labels_preprocessed_csv_path)

    print("Data preprocessing and subsetting completed successfully!")

    # ------------------------------
    # 8) Get a Subset of the Dataset
    # ------------------------------
    # Load edge_list.csv with columns ['src', 'dst']
    edges_df = pd.read_csv(edges_preprocessed_csv_path)

    # Load paper_info.csv with columns ['node_id', 'title', 'abstract']
    papers_df = pd.read_csv(papers_preprocessed_csv_path)

    # Load labels.csv with columns ['node_id', 'label']
    labels_df = pd.read_csv(labels_preprocessed_csv_path)

    # ------------------------------
    # 1.1) Verify Data Types and Consistency
    # ------------------------------
    # Ensure that 'src' and 'dst' in edges_df are of the same type as 'node_id' in papers_df and labels_df
    # Convert all node identifiers to string for consistency (or choose another consistent type)
    edges_df['src'] = edges_df['src'].astype(str)
    edges_df['dst'] = edges_df['dst'].astype(str)
    papers_df['node_id'] = papers_df['node_id'].astype(str)
    labels_df['node_id'] = labels_df['node_id'].astype(str)

    # ------------------------------
    # 2) Build Graph Using NetworkX
    # ------------------------------
    # Create an undirected graph to identify connected components.
    G = nx.from_pandas_edgelist(
        edges_df,
        source='src',
        target='dst',
        create_using=nx.Graph()  # Create an undirected graph
    )

    # ------------------------------
    # 3) Find the Largest Connected Component
    # ------------------------------
    components = nx.connected_components(G)  # This function returns all connected components
    largest_component = max(components, key=len)  # Select the component with the most nodes

    # ------------------------------
    # 4) Sample Nodes from the Largest Component
    # ------------------------------
    desired_size = 57471  # Adjust this value as needed
    largest_component = list(largest_component)  # Convert from set to list for easy sampling

    if len(largest_component) > desired_size:
        # If the component has more than desired_size nodes, randomly sample desired_size nodes
        sampled_nodes = random.sample(largest_component, desired_size)
        print(f"Randomly sampled {desired_size} nodes from the largest connected component.")
    else:
        # Otherwise, take all nodes in the component
        sampled_nodes = largest_component
        print(f"The largest connected component has {len(largest_component)} nodes, which is less than or equal to the desired size.")

    # Convert sampled_nodes to a set for efficient lookup
    sampled_nodes_set = set(sampled_nodes)

    # ------------------------------
    # 5) Filter CSVs Based on Selected Nodes
    # ------------------------------
    # 5.1) Filter Edges
    # Only include edges where both src and dst are in sampled_nodes
    edges_sub_df = edges_df[
        (edges_df['src'].isin(sampled_nodes_set)) &
        (edges_df['dst'].isin(sampled_nodes_set))
    ].reset_index(drop=True)

    # 5.2) Filter Paper Information
    papers_sub_df = papers_df[papers_df['node_id'].isin(sampled_nodes_set)].reset_index(drop=True)

    # 5.3) Filter Labels
    labels_sub_df = labels_df[labels_df['node_id'].isin(sampled_nodes_set)].reset_index(drop=True)

    # ------------------------------
    # 6) Verify Subsetting Results
    # ------------------------------
    # Check if the number of unique nodes in papers_sub_df and labels_sub_df matches the desired size
    unique_paper_nodes = papers_sub_df['node_id'].nunique()
    unique_label_nodes = labels_sub_df['node_id'].nunique()

    assert unique_paper_nodes == len(sampled_nodes_set), "Mismatch between sampled nodes and paper nodes."
    assert unique_label_nodes == len(sampled_nodes_set), "Mismatch between sampled nodes and label nodes."

    print("Verification passed: Sampled nodes match the papers and labels.")

    # ------------------------------
    # 7) Save the Subsets to New CSV Files
    # ------------------------------
    edges_sub_df.to_csv("data/processed/edges_sub.csv", index=False)
    papers_sub_df.to_csv("data/processed/papers_sub.csv", index=False)
    labels_sub_df.to_csv("data/processed/labels_sub.csv", index=False)

    print("Finished sampling!")
    print(f"Selected {len(sampled_nodes)} nodes (papers).")
    print(f"Subgraph edges shape: {edges_sub_df.shape}")
    print(f"Papers subset shape: {papers_sub_df.shape}")
    print(f"Labels subset shape: {labels_sub_df.shape}")

if __name__ == "__main__":
    main()
