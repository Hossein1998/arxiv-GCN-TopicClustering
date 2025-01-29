import string
import nltk
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ------------------------------
# Preprocessing function to clean and tokenize text
# ------------------------------
def preprocess_text(texts):
    """
    Preprocess input texts by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Tokenizing the text
    4. Removing stopwords
    5. Lemmatizing words

    Args:
        texts (list of str): List of input text documents.

    Returns:
        list of list: List of tokenized and cleaned texts.
    """
    stop_words = set(stopwords.words('english'))  # Set of stopwords to filter out
    lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer
    processed_texts = []  # List to store processed texts
    
    # Iterate over each text document
    for idx, text in enumerate(texts):
        try:
            # Convert text to lowercase
            text = text.lower()
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Tokenize the text into words
            tokens = text.split()
            # Remove stopwords and lemmatize the remaining words
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
            processed_texts.append(tokens)  # Append the processed tokens
        except Exception as e:
            print(f"Error processing text at index {idx}: {e}")
            # Append an empty list in case of an error
            processed_texts.append([])

    return processed_texts

# ------------------------------
# Clustering Evaluation Functions
# ------------------------------
def evaluate_clustering(true_labels, cluster_labels, embeddings):
    """
    Evaluate clustering performance using various metrics.

    Args:
        true_labels (list or array): Ground truth labels.
        cluster_labels (list or array): Predicted cluster labels.
        embeddings (numpy.ndarray): Embeddings used for clustering.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {}
    try:
        sil_score = silhouette_score(embeddings, cluster_labels)
        metrics['Silhouette Score'] = sil_score
    except Exception as e:
        metrics['Silhouette Score'] = None
        print(f"Silhouette Score computation failed: {e}")

    try:
        db_index = davies_bouldin_score(embeddings, cluster_labels)
        metrics['Davies-Bouldin Index'] = db_index
    except Exception as e:
        metrics['Davies-Bouldin Index'] = None
        print(f"Davies-Bouldin Index computation failed: {e}")

    try:
        ari = adjusted_rand_score(true_labels, cluster_labels)
        metrics['Adjusted Rand Index (ARI)'] = ari
    except Exception as e:
        metrics['Adjusted Rand Index (ARI)'] = None
        print(f"Adjusted Rand Index (ARI) computation failed: {e}")

    try:
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        metrics['Normalized Mutual Information (NMI)'] = nmi
    except Exception as e:
        metrics['Normalized Mutual Information (NMI)'] = None
        print(f"Normalized Mutual Information (NMI) computation failed: {e}")

    return metrics

def save_metrics(metrics, save_path):
    """
    Save evaluation metrics to a pickle file.

    Args:
        metrics (dict): Dictionary of evaluation metrics.
        save_path (str): Path to save the metrics file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Metrics saved to {save_path}")

def save_cluster_labels(cluster_labels, save_path):
    """
    Save cluster labels to a pickle file.

    Args:
        cluster_labels (list or array): Cluster labels.
        save_path (str): Path to save the cluster labels file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(cluster_labels, f)
    print(f"Cluster labels saved to {save_path}")

def visualize_clusters(embeddings, cluster_labels, num_clusters, title, save_path):
    """
    Visualize clustering results using t-SNE.

    Args:
        embeddings (numpy.ndarray): Embeddings used for clustering.
        cluster_labels (list or array): Cluster labels.
        num_clusters (int): Number of clusters.
        title (str): Title for the plot.
        save_path (str): Path to save the plot image.
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=cluster_labels,
        cmap='tab20',
        alpha=0.7,
        s=10
    )

    # Create legend for clusters
    handles, _ = scatter.legend_elements(num=num_clusters)
    labels = [f"Cluster {i}" for i in range(num_clusters)]
    plt.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Clustering visualization saved to {save_path}")
