# src/kmeans_clustering.py

import os
import sys
import torch
import pandas as pd
from sklearn.cluster import KMeans
from utils import evaluate_clustering, save_metrics, save_cluster_labels, visualize_clusters
from gnn_training import GCN
import warnings

def get_project_root() -> str:
    """
    Returns the absolute path to the project root directory.
    
    Returns:
        str: The absolute path to the project root.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

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
    # 1. Setup Paths and Environment
    # ===============================
    
    # Get project root
    project_root = get_project_root()
    
    # Add project root to sys.path to allow imports
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # Define paths relative to project root
    model_path = os.path.join(project_root, "models", "gcn_model.pth")
    data_pt_path = os.path.join(project_root, "data", "processed", "data.pt")
    labels_csv_path = os.path.join(project_root, "data", "processed", "labels_sub.csv")
    papers_csv_path = os.path.join(project_root, "data", "processed", "papers_sub.csv")
    metrics_save_path = os.path.join(project_root, "results", "metrics", "clustering_metrics.pkl")
    cluster_labels_save_path = os.path.join(project_root, "results", "metrics", "cluster_labels.pkl")
    visualization_save_path = os.path.join(project_root, "results", "plots", "kmeans_tsne.png")
    
    # ایجاد دایرکتوری‌های مورد نیاز
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(cluster_labels_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(visualization_save_path), exist_ok=True)
    
    # ===============================
    # 2. Load the GCN Model and Data
    # ===============================
    
    # بررسی وجود مدل GCN
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GCN model not found at {model_path}")
    
    # بررسی وجود فایل داده
    if not os.path.exists(data_pt_path):
        raise FileNotFoundError(f"Data file not found at {data_pt_path}")
    
    # بارگذاری داده‌ها با نادیده گرفتن FutureWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        data = torch.load(data_pt_path)
    
    # بارگذاری مدل GCN
    num_classes = data.y.max().item() + 1  # باید 39 باشد
    input_dim = data.num_features
    hidden_dim = 512
    output_dim = num_classes  # به عنوان تعداد خوشه‌ها استفاده می‌شود
    dropout = 0.5
    
    model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # ===============================
    # 3. Extract Embeddings
    # ===============================
    
    embeddings = get_hidden_embeddings(model, data)
    print(f"Hidden Embeddings Shape: {embeddings.shape}")  # انتظار: (num_nodes, hidden_dim)
    
    # ===============================
    # 4. Apply K-Means Clustering
    # ===============================
    
    num_clusters = output_dim  # تعداد خوشه‌ها برابر با تعداد کلاس‌ها
    print(f"Number of clusters: {num_clusters}")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    print("K-Means clustering completed.")
    
    # ===============================
    # 5. Evaluate Clustering Quality
    # ===============================
    
    # بررسی وجود فایل‌های CSV
    if not os.path.exists(labels_csv_path):
        raise FileNotFoundError(f"Labels CSV file not found at {labels_csv_path}")
    if not os.path.exists(papers_csv_path):
        raise FileNotFoundError(f"Papers CSV file not found at {papers_csv_path}")
    
    # بارگذاری برچسب‌های واقعی
    df_labels = pd.read_csv(labels_csv_path)
    df_papers = pd.read_csv(papers_csv_path)
    node_order = df_papers['node_id'].tolist()
    
    # اطمینان از تطابق ترتیب گره‌ها
    true_labels = df_labels.set_index('node_id').loc[node_order]['label'].tolist()
    
    # بررسی تطابق تعداد نمونه‌ها
    if len(true_labels) != len(cluster_labels):
        print(f"Mismatch in number of samples: true_labels has {len(true_labels)}, cluster_labels has {len(cluster_labels)}")
        # برش به حداقل طول
        common_length = min(len(true_labels), len(cluster_labels))
        true_labels = true_labels[:common_length]
        cluster_labels = cluster_labels[:common_length]
    
    # ارزیابی خوشه‌بندی
    metrics = evaluate_clustering(true_labels, cluster_labels, embeddings)
    print("\n--- K-Means Clustering Evaluation Metrics ---")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: Not computed.")
    
    # ذخیره متریک‌ها
    save_metrics(metrics, metrics_save_path)
    
    # ذخیره برچسب‌های خوشه‌بندی شده
    save_cluster_labels(cluster_labels, cluster_labels_save_path)
    
    # ===============================
    # 6. Visualize Clustering Results with t-SNE
    # ===============================
    
    print("\nVisualizing K-Means Clustering Results with t-SNE...")
    
    # کاهش ابعاد با t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # تجسم با استفاده از تابع یوتیلیتی
    visualize_clusters(
        embeddings=embeddings_2d,
        cluster_labels=cluster_labels,
        num_clusters=num_clusters,
        title='t-SNE Visualization of K-Means Clusters',
        save_path=visualization_save_path
    )
    
    print(f"K-Means t-SNE visualization saved to {visualization_save_path}")

if __name__ == "__main__":
    main()
