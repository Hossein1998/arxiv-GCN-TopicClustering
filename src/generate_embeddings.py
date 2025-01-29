import os
import torch
import pandas as pd
import numpy as np
import gdown
from typing import List
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch_geometric.data import Data
from huggingface_hub import login  # Hugging Face authentication

DATA_FILE_ID = "1Xkrz7nI9vAfIN0Ij2T_qLL0V4F9ORRgq"
DATA_URL = f"https://drive.google.com/uc?id={DATA_FILE_ID}"
DATA_PATH = "data/processed/data.pt"

current_path = os.getcwd()
if current_path.endswith("src"):
    os.chdir("..")  # Move one directory up to the main project folder
print(f"Current working directory: {os.getcwd()}")  # Confirm the current directory

HF_TOKEN = os.getenv("HF_TOKEN")

def download_data_file():
    """
    Download the 'data.pt' file from Google Drive in case of errors.
    """
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    print(f"Downloading data.pt from Google Drive to {DATA_PATH}...")
    gdown.download(DATA_URL, DATA_PATH, quiet=False)
    print(f"Download complete: {DATA_PATH}")

if not HF_TOKEN:
    print("Error: HF_TOKEN is missing! Downloading 'data.pt' instead...")
    download_data_file()
    exit()

login(token=HF_TOKEN)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

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

def get_embeddings_with_resume(
    texts: List[str],
    tokenizer,
    model,
    batch_size: int = 8,
    max_length: int = 2048,
    device: torch.device = torch.device("cpu"),
    temp_path: str = "data/processed/partial_embeddings.pt"
) -> torch.Tensor:
    """
    Generate embeddings for a list of texts with resume capability.

    Args:
        texts (List[str]): List of input texts.
        tokenizer: Hugging Face tokenizer.
        model: Pretrained Hugging Face model.
        batch_size (int): Number of texts per batch.
        max_length (int): Maximum token length per text.
        device (torch.device): Device to run the model on.
        temp_path (str): Path to save partial embeddings.

    Returns:
        torch.Tensor: Concatenated embeddings for all texts.
    """
    partial_embeddings = []
    start_index = 0

    # Load existing partial embeddings if available
    if os.path.exists(temp_path):
        partial_embeddings = torch.load(temp_path)
        start_index = len(partial_embeddings)
        print(f"Found existing partial embeddings with {start_index} batches.")
    else:
        print("No existing partial embeddings found. Starting fresh.")

    total_batches = (len(texts) + batch_size - 1) // batch_size

    # Generate embeddings batch by batch
    for batch_i in tqdm(range(start_index, total_batches), desc="Generating embeddings", unit="batch"):
        start = batch_i * batch_size
        end = start + batch_size
        batch_texts = texts[start:end]

        # Tokenize the batch of texts
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the mean of the last hidden state as the embedding
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()

        # Append embeddings of the current batch to the list
        partial_embeddings.append(embeddings)

        # Save partial embeddings to file
        torch.save(partial_embeddings, temp_path)

    # Concatenate all embeddings
    all_embeddings = torch.cat(partial_embeddings, dim=0)
    return all_embeddings

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
