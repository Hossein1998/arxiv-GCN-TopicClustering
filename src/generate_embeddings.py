import os
import torch
import pandas as pd
import numpy as np
import pickle
from typing import List
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

def load_model(model_name: str = MODEL_NAME):
    """
    Loads the tokenizer and model for a LLaMa-based (or any other HF) model into memory.
    
    Args:
        model_name (str): The name of the pretrained model.
    
    Returns:
        tokenizer (transformers.PreTrainedTokenizer)
        model (transformers.PreTrainedModel)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # If pad_token is not defined for this tokenizer, set it to eos_token
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
    temp_path: str = "partial_embeddings.pt"
) -> torch.Tensor:
    """
    Generates embeddings for a list of texts with resume capability in case of interruptions.
    
    Args:
        texts (List[str]): List of input texts.
        tokenizer: Tokenizer corresponding to the model.
        model: Pretrained language model.
        batch_size (int): Number of texts per batch.
        max_length (int): Maximum token length per text.
        device (torch.device): Device to run the model on.
        temp_path (str): Path to save partial embeddings.
    
    Returns:
        torch.Tensor: Concatenated embeddings for all texts.
    """
    partial_embeddings = []
    start_index = 0

    # If a temporary results file already exists, load it
    if os.path.exists(temp_path):
        partial_embeddings = torch.load(temp_path)
        start_index = len(partial_embeddings)
        print(f"Found existing partial embeddings with {start_index} batches.")
    else:
        print("No existing partial embeddings found. Starting fresh.")

    total_batches = (len(texts) + batch_size - 1) // batch_size

    # Start the loop from where it was previously left off
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

        with torch.no_grad():
            outputs = model(**inputs)
            # Use the mean of the last hidden state as the embedding
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()

        # Add the embeddings of this batch to the list
        partial_embeddings.append(embeddings)

        # Save the temporary results after each iteration
        torch.save(partial_embeddings, temp_path)

    # Finally, concatenate all batches together
    all_embeddings = torch.cat(partial_embeddings, dim=0)
    return all_embeddings

def main():
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
        temp_path=temp_path  # Temporary storage path
    )
    print("Embeddings generation completed.")

    # ---------------------------------------
    # 4) Save Embeddings
    # ---------------------------------------
    torch.save(embeddings, embeddings_save_path)
    print(f"Embeddings saved to {embeddings_save_path}")

    # ---------------------------------------
    # 5) Load and Process Edge List
    # ---------------------------------------
    edges_sub_csv_path = "data/processed/edges_sub.csv"
    edge_df = pd.read_csv(edges_sub_csv_path)
    
    # Mapping node_ids to integer indices
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
    # 6) Define Labels
    # ---------------------------------------
    labels_sub_csv_path = "data/processed/labels_sub.csv"
    labels_df = pd.read_csv(labels_sub_csv_path)
    
    # Ensure the order of labels matches the node order
    labels_df = labels_df.set_index('node_id').loc[unique_nodes].reset_index()
    labels = labels_df["label"].astype(int).tolist()
    y = torch.tensor(labels, dtype=torch.long)
    print(f"Labels tensor shape: {y.shape}")

    # ---------------------------------------
    # 7) Create a Data Object from PyTorch Geometric
    # ---------------------------------------
    data = Data(
        x=embeddings,           # Node features
        edge_index=edge_index,  # Edge indices
        y=y,                    # Labels
        num_nodes=len(unique_nodes)
    )
    print("Data object created successfully:")
    print(data)

    # ---------------------------------------
    # 8) Save the Data Object
    # ---------------------------------------
    data_save_path = "data/processed/data.pt"
    torch.save(data, data_save_path)
    print(f"Data object saved to {data_save_path}")

if __name__ == "__main__":
    main()
