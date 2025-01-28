import pickle
import matplotlib.pyplot as plt
import os

def load_metrics(metric_path):
    """
    Load clustering metrics from a pickle file.
    """
    with open(metric_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

def main():
    print("\nStarting Result Analysis...\n")
    
    # Paths to metrics
    gcn_metrics_path = "results/metrics/clustering_metrics.pkl"
    lda_metrics_path = "results/metrics/lda_clustering_metrics.pkl"
    
    # Load metrics
    if os.path.exists(gcn_metrics_path):
        gcn_metrics = load_metrics(gcn_metrics_path)
        print("GCN-based K-Means Clustering Metrics:")
        for metric, value in gcn_metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: Not computed.")
    else:
        print(f"GCN-based metrics file not found at {gcn_metrics_path}")
        gcn_metrics = {}
    
    if os.path.exists(lda_metrics_path):
        lda_metrics = load_metrics(lda_metrics_path)
        print("\nLDA-based Clustering Metrics:")
        for metric, value in lda_metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: Not computed.")
    else:
        print(f"LDA-based metrics file not found at {lda_metrics_path}")
        lda_metrics = {}
    
    # Comparison
    comparison_metrics = ['Silhouette Score', 'Davies-Bouldin Index', 'Adjusted Rand Index (ARI)', 'Normalized Mutual Information (NMI)']
    gcn_values = [gcn_metrics.get(metric, None) for metric in comparison_metrics]
    lda_values = [lda_metrics.get(metric, None) for metric in comparison_metrics]
    
    x = range(len(comparison_metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar([p - width/2 for p in x], gcn_values, width, label='GCN-based K-Means')
    plt.bar([p + width/2 for p in x], lda_values, width, label='LDA-based Grouping')
    
    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Scores')
    plt.title('Comparison of Clustering Evaluation Metrics')
    plt.xticks(x, comparison_metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot_path = "results/plots/clustering_comparison.png"
    os.makedirs(os.path.dirname(comparison_plot_path), exist_ok=True)
    plt.savefig(comparison_plot_path, dpi=300)
    plt.show()
    print(f"Clustering comparison plot saved to {comparison_plot_path}")

if __name__ == "__main__":
    main()
