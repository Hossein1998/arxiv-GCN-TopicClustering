import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
import gdown
import warnings

# ===============================
# 1. Utility Functions
# ===============================

def get_project_root() -> str:
    """
    Returns the absolute path to the project root directory.
    
    Returns:
        str: The absolute path to the project root.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def download_data_file(data_url: str, data_path: str):
    """
    Downloads the 'data.pt' file from Google Drive if it is missing.
    
    Args:
        data_url (str): The direct download URL from Google Drive.
        data_path (str): The local path where the data will be saved.
    """
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    print(f"Downloading data.pt from Google Drive to {data_path}...")
    gdown.download(data_url, data_path, quiet=False)
    print(f"Download complete: {data_path}")

def load_data(data_path: str) -> Data:
    """
    Loads the dataset from the specified path. If the dataset is not found, attempts to download it.
    
    Args:
        data_path (str): The path to the 'data.pt' file.
    
    Returns:
        Data: The PyTorch Geometric data object.
    """
    if not os.path.exists(data_path):
        print("data.pt not found. Please ensure the file exists or modify the DATA_FILE_ID and DATA_URL.")
        raise FileNotFoundError(f"The file '{data_path}' does not exist.")
    
    # Suppress FutureWarning for torch.load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        data = torch.load(data_path)
    
    return data

def display_data_info(data: Data):
    """
    Displays information about the loaded dataset.
    
    Args:
        data (Data): The PyTorch Geometric data object.
    """
    print(f"\nDataset Information:")
    print(f"Number of Nodes: {data.num_nodes}")
    print(f"Number of Features: {data.num_features}")
    print(f"Number of Classes: {data.y.max().item() + 1}")
    print(f"Edge Index Shape: {data.edge_index.shape}")
    print(f"Labels Shape: {data.y.shape}")

def adjust_labels(data: Data) -> Data:
    """
    Ensures that the labels start from 0.
    
    Args:
        data (Data): The PyTorch Geometric data object.
    
    Returns:
        Data: The updated data object with adjusted labels.
    """
    if data.y.min() == 1:
        data.y = data.y - 1
        print(f"Labels adjusted to start from 0. New label range: {data.y.min()} to {data.y.max()}")
    else:
        print(f"Label range: {data.y.min()} to {data.y.max()}")
    return data

# ===============================
# 2. Model Definition
# ===============================

class GCN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5):
        """
        Initializes the Graph Convolutional Network (GCN).
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            output_dim (int): Number of output classes.
            dropout (float): Dropout probability.
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data: Data):
        """
        Defines the forward pass of the GCN.
        
        Args:
            data (Data): The PyTorch Geometric data object.
        
        Returns:
            Tuple[Tensor, Tensor]: Output logits and hidden embeddings.
        """
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        hidden = x  # Hidden layer embeddings
        x = self.conv2(x, edge_index)

        return x, hidden

# ===============================
# 3. Data Splitting
# ===============================

def train_val_test_split(data: Data, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2, num_classes: int = 40) -> Data:
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
        if idx.numel() == 0:
            continue  # Skip if no samples for the class
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

# ===============================
# 4. Data Integrity Checks and Fix
# ===============================

def check_and_fix_edge_indices(data: Data, num_classes: int) -> Data:
    """
    Ensures that all edge indices are within the valid range of node indices.
    Removes any invalid edges that reference non-existent nodes.
    
    Args:
        data (Data): The PyTorch Geometric data object.
        num_classes (int): Number of unique classes.
    
    Returns:
        Data: The updated data object with valid edge indices.
    """
    max_edge_index = data.edge_index.max().item()
    print(f"Max node index in edge_index: {max_edge_index}")
    print(f"Number of nodes: {data.num_nodes}")

    # Verify if any edge references a node index >= num_nodes
    if max_edge_index >= data.num_nodes:
        print(f"Found edge indices with node indices >= {data.num_nodes}. Removing these edges.")
        # Identify invalid edges
        invalid_mask = (data.edge_index[0] >= data.num_nodes) | (data.edge_index[1] >= data.num_nodes)
        num_invalid = invalid_mask.sum().item()
        print(f"Number of invalid edges: {num_invalid}")

        # Remove invalid edges
        data.edge_index = data.edge_index[:, ~invalid_mask]
        print(f"New edge_index shape: {data.edge_index.shape}")
    else:
        print("All edge indices are within the valid range.")

    # Ensure labels are in correct range
    assert data.y.min().item() >= 0, f"Minimum label is {data.y.min().item()}, should be >=0"
    assert data.y.max().item() < num_classes, f"Maximum label is {data.y.max().item()}, should be < {num_classes}"

    # Ensure edge indices are valid
    assert data.edge_index.min().item() >= 0, "Negative edge indices found."
    assert data.edge_index.max().item() < data.num_nodes, "Edge indices exceed number of nodes."

    # Ensure labels are of type long
    if data.y.dtype != torch.long:
        data.y = data.y.long()

    return data

# ===============================
# 5. Training and Evaluation Functions
# ===============================

def train(model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> float:
    """
    Trains the GCN model for one epoch.
    
    Args:
        model (torch.nn.Module): The GCN model.
        data (Data): The PyTorch Geometric data object.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
    
    Returns:
        float: The training loss.
    """
    model.train()
    optimizer.zero_grad()
    out, _ = model(data)  # Unpack the two outputs
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model: torch.nn.Module, data: Data, mask: torch.Tensor) -> float:
    """
    Evaluates the model on the specified mask.
    
    Args:
        model (torch.nn.Module): The GCN model.
        data (Data): The PyTorch Geometric data object.
        mask (torch.Tensor): The mask indicating which nodes to evaluate.
    
    Returns:
        float: The accuracy on the specified mask.
    """
    model.eval()
    with torch.no_grad():
        out, _ = model(data)  # Unpack the two outputs
        logits = out[mask]
        labels = data.y[mask]
        pred = logits.argmax(dim=1)
        correct = (pred == labels).sum().item()
        acc = correct / mask.sum().item()
    return acc

def get_hidden_embeddings(model: torch.nn.Module, data: Data) -> np.ndarray:
    """
    Extracts the hidden layer embeddings from the model.
    
    Args:
        model (torch.nn.Module): The GCN model.
        data (Data): The PyTorch Geometric data object.
    
    Returns:
        np.ndarray: The hidden layer embeddings.
    """
    model.eval()
    with torch.no_grad():
        _, hidden = model(data)
    return hidden.cpu().numpy()

# ===============================
# 6. Main Function
# ===============================

def main():
    """
    Runs the complete GCN model training and evaluation process.
    """
    # Get project root
    project_root = get_project_root()
    
    # Define paths relative to project root
    DATA_FILE_ID = "1Xkrz7nI9vAfIN0Ij2T_qLL0V4F9ORRgq"
    DATA_URL = f"https://drive.google.com/uc?id={DATA_FILE_ID}"
    DATA_PATH = os.path.join(project_root, "data", "processed", "data.pt")
    MODELS_DIR = os.path.join(project_root, "models")
    RESULTS_DIR = os.path.join(project_root, "results", "plots")
    
    # Download data.pt if it doesn't exist
    if not os.path.exists(DATA_PATH):
        print("data.pt not found. Downloading from Google Drive...")
        download_data_file(DATA_URL, DATA_PATH)

    # Load dataset
    data = load_data(DATA_PATH)

    # Display dataset information
    display_data_info(data)

    # Adjust labels if necessary
    data = adjust_labels(data)

    # Apply train/val/test split
    num_classes = data.y.max().item() + 1
    print(f"Number of classes after adjustment: {num_classes}")
    data = train_val_test_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, num_classes=num_classes)

    # Data integrity checks
    data = check_and_fix_edge_indices(data, num_classes)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Define model parameters
    input_dim = data.num_features      # e.g., 2048
    hidden_dim = 512
    output_dim = num_classes           # e.g., 39
    dropout = 0.5

    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Output dimension: {output_dim}")

    # Initialize the model, optimizer, and loss function
    model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Training parameters
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
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save the best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_DIR, "gcn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Best GCN model saved to {model_save_path}")

    # Evaluate the model on the test set
    test_acc = evaluate(model, data, data.test_mask)
    print(f'\nTest Accuracy: {test_acc:.4f}')

    # ===============================
    # 7. Evaluation with t-SNE Visualization
    # ===============================
    
    print("\nGenerating embeddings and visualizing with t-SNE...")
    
    # Extract hidden embeddings
    hidden_embeddings = get_hidden_embeddings(model, data)
    
    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(hidden_embeddings)
    
    # Plotting
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=data.y.cpu().numpy(),
        cmap='tab20',
        alpha=0.7,
        s=10
    )
    
    # Create a legend for up to num_classes classes
    handles, _ = scatter.legend_elements(num=num_classes)
    labels = [f"Class {i}" for i in range(num_classes)]
    plt.legend(handles, labels, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.title('t-SNE Visualization of GCN Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_save_path = os.path.join(RESULTS_DIR, "gcn_tsne.png")
    plt.savefig(plot_save_path, dpi=300)
    print(f"t-SNE plot saved to {plot_save_path}")

if __name__ == "__main__":
    main()
