import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import save_metrics, save_cluster_labels

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
        hidden = x  # Hidden layer embeddings
        x = self.conv2(x, edge_index)

        return x, hidden

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data)  # Output from final layer and hidden layer
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
    # ===============================
    # 1. Loading the Data
    # ===============================

    # Define the path to your data.pt file
    data_path = 'data/processed/data.pt'

    # Check if the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file '{data_path}' does not exist. Please ensure the path is correct.")

    # Load the data with caution regarding the FutureWarning
    try:
        data = torch.load(data_path)
    except FutureWarning as e:
        print(e)
        # In future versions, you might need to set weights_only=True
        data = torch.load(data_path, weights_only=True)

    # Verify the data object
    print(f"\nNumber of Nodes: {data.num_nodes}")
    print(f"Number of Features: {data.num_features}")
    print(f"Number of Classes: {data.y.max().item() + 1}")
    print(f"Edge Index Shape: {data.edge_index.shape}")
    print(f"Labels Shape: {data.y.shape}")

    # Ensure labels start from 0
    if data.y.min() == 1:
        data.y = data.y - 1

    print(f"Label range after adjustment: {data.y.min()} to {data.y.max()}")

    # ===============================
    # 2. Data Splitting
    # ===============================

    num_classes = data.y.max().item() + 1  # Should be 39 based on adjusted labels
    print(f"Number of classes after adjustment: {num_classes}")
    data = train_val_test_split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, num_classes=num_classes)

    # ===============================
    # 3. Data Integrity Checks and Fix
    # ===============================

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Check the maximum node index in edge_index
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

    # Move data to device
    data = data.to(device)

    # ===============================
    # 4. Model Initialization
    # ===============================

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

    # ===============================
    # 5. Training the GCN
    # ===============================

    # Training parameters
    epochs = 200
    best_val_acc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    print("\nStarting training...\n")

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

    # ===============================
    # 6. Testing the Model
    # ===============================

    test_acc = evaluate(model, data, data.test_mask)
    print(f'\nTest Accuracy: {test_acc:.4f}')

    # ===============================
    # 7. Embedding Extraction and t-SNE Visualization
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

    # Create a legend for up to 40 classes
    handles, _ = scatter.legend_elements(num=output_dim)
    labels = [f"Class {i}" for i in range(output_dim)]
    plt.legend(handles, labels, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.title('t-SNE Visualization of GCN Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plot_save_path = "results/plots/gcn_tsne.png"
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path, dpi=300)
    plt.show()
    print(f"t-SNE plot saved to {plot_save_path}")

    # ===============================
    # 8. Save the Best Model and Embeddings
    # ===============================

    # Save the best model
    model_save_path = "models/gcn_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Best GCN model saved to {model_save_path}")

    # Save embeddings
    embeddings_save_path = "models/hidden_embeddings.pt"
    torch.save(hidden_embeddings, embeddings_save_path)
    print(f"Hidden embeddings saved to {embeddings_save_path}")

if __name__ == "__main__":
    main()
