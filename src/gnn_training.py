import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from utils import save_metrics, save_cluster_labels
import gdown

# Change working directory if running from src/
current_path = os.getcwd()
if current_path.endswith("src"):
    os.chdir("..")  
print(f"Current working directory: {os.getcwd()}")  

# Google Drive file details for backup
DATA_FILE_ID = "1Xkrz7nI9vAfIN0Ij2T_qLL0V4F9ORRgq"
DATA_URL = f"https://drive.google.com/uc?id={DATA_FILE_ID}"
DATA_PATH = "data/processed/data.pt"

def download_data_file():
    """
    Downloads the 'data.pt' file from Google Drive if it is missing.
    """
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    print(f"Downloading data.pt from Google Drive to {DATA_PATH}...")
    gdown.download(DATA_URL, DATA_PATH, quiet=False)
    print(f"Download complete: {DATA_PATH}")

# Check if data.pt exists, otherwise download it
if not os.path.exists(DATA_PATH):
    print("data.pt not found. Downloading from Google Drive...")
    download_data_file()

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        hidden = x  
        x = self.conv2(x, edge_index)

        return x, hidden

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out, _ = model(data)
        logits = out[mask]
        labels = data.y[mask]
        pred = logits.argmax(dim=1)
        correct = (pred == labels).sum().item()
        acc = correct / mask.sum().item()
    return acc

def get_hidden_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        _, hidden = model(data)
    return hidden.cpu().numpy()

def train_val_test_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, num_classes=40):
    """
    Splits the dataset into training, validation, and test masks.

    Args:
        data (Data): The PyTorch Geometric data object.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
        test_ratio (float): Proportion of data for testing.
        num_classes (int): Number of unique classes.

    Returns:
        Data: The data object with updated masks.
    """
    # Initialize masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    # For each class, split the nodes
    for c in range(num_classes):
        idx = (data.y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.size(0))]  # Shuffle

        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        n_test = len(idx) - n_train - n_val

        if n_train == 0:
            n_train = 1  # Ensure at least one training sample per class
        if n_val == 0 and n_test > 0:
            n_val = 1
        if n_test == 0 and n_val > 1:
            n_test = 1

        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

def main():
    """
    Runs the complete GCN model training and evaluation process.
    """

    # Ensure necessary directories exist
    processed_dir = "data/processed"
    models_dir = "models"
    results_dir = "results/plots"
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load dataset
    try:
        data = torch.load(DATA_PATH)
    except Exception as e:
        print(f"Error loading data.pt: {e}")
        print("Downloading data.pt from Google Drive...")
        download_data_file()
        data = torch.load(DATA_PATH)  

    # Ensure labels are in correct range
    if data.y.min() == 1:
        data.y = data.y - 1  # Adjust labels if they start from 1

    num_classes = data.y.max().item() + 1
    print(f"\nDataset Information:")
    print(f"Number of Nodes: {data.num_nodes}")
    print(f"Number of Features: {data.num_features}")
    print(f"Number of Classes: {num_classes}")
    print(f"Edge Index Shape: {data.edge_index.shape}")
    print(f"Labels Shape: {data.y.shape}")

    # Apply train/val/test split
    data = train_val_test_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, num_classes=num_classes)

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Define model parameters
    input_dim = data.num_features
    hidden_dim = 512
    output_dim = num_classes
    dropout = 0.5

    model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    epochs = 200
    best_val_acc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    print("\nStarting GCN training...\n")

    for epoch in range(1, epochs + 1):
        loss = train(model, data, optimizer, criterion)
        train_acc = evaluate(model, data, data.train_mask)
        val_acc = evaluate(model, data, data.val_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')

    # Save the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_save_path = os.path.join(models_dir, "gcn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Best GCN model saved to {model_save_path}")

    # Evaluate the model on the test set
    test_acc = evaluate(model, data, data.test_mask)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # Generate embeddings and visualize with t-SNE
    print("\nGenerating embeddings and visualizing with t-SNE...")

    hidden_embeddings = get_hidden_embeddings(model, data)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(hidden_embeddings)

    # Create the t-SNE plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=data.y.cpu().numpy(), cmap='tab20', alpha=0.7, s=10)
    handles, _ = scatter.legend_elements(num=output_dim)
    labels = [f"Class {i}" for i in range(output_dim)]
    plt.legend(handles, labels, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.title('t-SNE Visualization of GCN Embeddings')
    plt.tight_layout()

    plot_save_path = os.path.join(results_dir, "gcn_tsne.png")
    plt.savefig(plot_save_path, dpi=300)
    plt.show()
    print(f"t-SNE plot saved to {plot_save_path}")

if __name__ == "__main__":
    main()
