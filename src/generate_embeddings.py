import os
import torch
import pandas as pd
import numpy as np
import gdown
from typing import List
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch_geometric.data import Data
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Change to the parent directory if the script is running from the src folder
current_path = os.getcwd()
if current_path.endswith("src"):
    os.chdir("..")  # Move one directory up to the main project folder
print(f"Current working directory: {os.getcwd()}")  # Confirm the current directory

# Retrieve the Hugging Face token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Define the model name
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

def download_data_file():
    """
    Download the 'data.pt' file from Google Drive in case of errors.
    """
    file_id = '1Xkrz7nI9vAfIN0Ij2T_qLL0V4F9ORRgq'
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "data/processed/data.pt"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Downloading data.pt from Google Drive to {output_path}...")
    gdown.download(url, output_path, quiet=False)
    print(f"Download complete: {output_path}")

def load_model(model_name: str = MODEL_NAME):
    """
    Load the tokenizer and model from Hugging Face.

    Args:
        model_name (str): The name of the pretrained model.

    Returns:
        tokenizer (transformers.AutoTokenizer): Tokenizer for the model.
        model (transformers.AutoModel): Pretrained model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModel.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    
    # Set pad_token to eos_token if it is not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

def main():
    """
    Main function to generate embeddings, process edges, and create a PyTorch Geometric data object.
    """

    # ---------------------------------------
    # 1) Ensure Processed Directory Exists
    # ---------------------------------------
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # ---------------------------------------
    # 2) Load the Model
    # ---------------------------------------
    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")

    # ---------------------------------------
    # 3) Load the Dataset
    # ---------------------------------------
    papers_csv_path = os.path.join(processed_dir, "papers_sub.csv")
    
    try:
        df = pd.read_csv(papers_csv_path)
        df['combined_text'] = df['title'].fillna('') + ". " + df['abstract'].fillna('')
        text_list = df['combined_text'].tolist()
        print(f"Number of texts to embed: {len(text_list)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Downloading 'data.pt' instead...")
        download_data_file()
        return  # Exit main function

    # ---------------------------------------
    # 4) Load Edge List
    # ---------------------------------------
    edges_sub_csv_path = os.path.join(processed_dir, "edges_sub.csv")

    try:
        edge_df = pd.read_csv(edges_sub_csv_path)
    except Exception as e:
        print(f"Error loading edge list: {e}")
        print("Downloading 'data.pt' instead...")
        download_data_file()
        return  # Exit main function

    # ---------------------------------------
    # 5) Map node_ids to integer indices
    # ---------------------------------------
    unique_nodes = sorted(df['node_id'].unique())
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
    print(f"Total unique nodes: {len(unique_nodes)}")

    # Convert edge list to integer indices
    edge_list = list(zip(edge_df['src'], edge_df['dst']))
    edge_index = torch.tensor(
        [[node_id_to_idx[src], node_id_to_idx[dst]] for src, dst in edge_list],
        dtype=torch.long
    ).t().contiguous()
    print(f"Edge index shape: {edge_index.shape}")

    # ---------------------------------------
    # 6) Save Edge Data
    # ---------------------------------------
    edge_index_path = os.path.join(processed_dir, "edge_index.pt")
    torch.save(edge_index, edge_index_path)
    print(f"Edge index saved to {edge_index_path}")

if __name__ == "__main__":
    main()
