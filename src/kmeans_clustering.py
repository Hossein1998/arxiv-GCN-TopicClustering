import torch
import pandas as pd
from sklearn.cluster import KMeans
from utils import evaluate_clustering, save_metrics, save_cluster_labels, visualize_clusters
import os

def get_hidden_embeddings(model, data):
    """
    Extract hidden embeddings from the GCN model.

    Args:
        model (torch.nn.Module): Trained GCN model.
        data (Data): PyTorch Geometric data object.

    Returns:
        numpy.ndarray: Extracted hidden embeddings.
    """
    model.eval()
    with torch.no_grad():
        _, hidden = model(data)
    return hidden.cpu().numpy()

def main():
    print("\nStarting K-Means Clustering...\n")
    
    # ===============================
    # 1. Load the GCN Model and Data
    # ===============================
    
    # Path to the saved GCN model
    model_path = "models/gcn_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GCN model not found at {model_path}")
    
    # Import GCN class from gnn_training.py
    from gnn_training import GCN
    
    # Load the data
    data_pt_path = "data/processed/data.pt"
    if not os.path.exists(data_pt_path):
        raise FileNotFoundError(f"Data file not found at {data_pt_path}")
    
    data = torch.load(data_pt_path)
    
    # Initialize the model
    input_dim = data.num_features
    hidden_dim = 512
    output_dim = data.y.max().item() + 1  # e.g., 39
    dropout = 0.5
    
    model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # ===============================
    # 2. Extract Embeddings
    # ===============================
    
    embeddings = get_hidden_embeddings(model, data)
    print(f"Hidden Embeddings Shape: {embeddings.shape}")  # Expected: (num_nodes, hidden_dim)
    
    # ===============================
    # 3. Apply K-Means Clustering
    # ===============================
    
    num_clusters = output_dim  # e.g., 39
    print(f"Number of clusters: {num_clusters}")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    print("K-Means clustering completed.")
    
    # ===============================
    # 4. Evaluate Clustering Quality
    # ===============================
    
    # Load true labels
    labels_csv_path = "data/processed/labels_sub.csv"
    papers_csv_path = "data/processed/papers_sub.csv"
    df_labels = pd.read_csv(labels_csv_path)
    df_papers = pd.read_csv(papers_csv_path)
    node_order = df_papers['node_id'].tolist()
    
    # Ensure alignment
    true_labels = df_labels.set_index('node_id').loc[node_order]['label'].tolist()
    
    if len(true_labels) != len(cluster_labels):
        print(f"Mismatch in number of samples: true_labels has {len(true_labels)}, cluster_labels has {len(cluster_labels)}")
        # Truncate to the minimum length
        common_length = min(len(true_labels), len(cluster_labels))
        true_labels = true_labels[:common_length]
        cluster_labels = cluster_labels[:common_length]
    
    # Evaluate clustering
    metrics = evaluate_clustering(true_labels, cluster_labels, embeddings)
    print("\n--- K-Means Clustering Evaluation Metrics ---")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: Not computed.")
    
    # Save metrics
    metrics_save_path = "results/metrics/clustering_metrics.pkl"
    save_metrics(metrics, metrics_save_path)
    
    # Save cluster labels
    cluster_labels_save_path = "results/metrics/cluster_labels.pkl"
    save_cluster_labels(cluster_labels, cluster_labels_save_path)
    
    # ===============================
    # 5. Visualize Clustering Results with t-SNE
    # ===============================
    
    print("\nVisualizing K-Means Clustering Results with t-SNE...")
    
    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Visualize using utility function
    visualization_save_path = "results/plots/kmeans_tsne.png"
    visualize_clusters(
        embeddings=embeddings_2d,
        cluster_labels=cluster_labels,
        num_clusters=num_clusters,
        title='t-SNE Visualization of K-Means Clusters',
        save_path=visualization_save_path
    )

if __name__ == "__main__":
    main()
