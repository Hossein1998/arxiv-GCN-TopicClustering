import os
import torch
import pandas as pd
import numpy as np
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
    os.chdir("..")  # Go one directory up
print(f"Current working directory: {os.getcwd()}")  # Print the current working directory

# Retrieve the Hugging Face token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Define the model name
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
    # 1) Load the Model
    # ---------------------------------------
    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")

    # ---------------------------------------
    # 2) Load the Dataset
    # ---------------------------------------
    papers_csv_path = "data/processed/papers_sub.csv"
    embeddings_save_path = "data/processed/embeddings.pt"
    temp_path = "data/processed/partial_embeddings.pt"

    df = pd.read_csv(papers_csv_path)
    df['combined_text'] = df['title'].fillna('') + ". " + df['abstract'].fillna('')
    text_list = df['combined_text'].tolist()
    print(f"Number of texts to embed: {len(text_list)}")

    # ---------------------------------------
    # 3) Get Embeddings with Resume Capability
    # ---------------------------------------
    embeddings = get_embeddings_with_resume(
        texts=text_list,
        tokenizer=tokenizer,
        model=model,
        batch_size=8,
        max_length=2048,
        device=device,
        temp_path=temp_path
    )
    print("Embeddings generation completed.")

    # ---------------------------------------
    # 4) Save Embeddings
    # ---------------------------------------
    torch.save(embeddings, embeddings_save_path)
    print(f"Embeddings saved to {embeddings_save_path}")

    # Remaining parts of the script...

if __name__ == "__main__":
    main()
