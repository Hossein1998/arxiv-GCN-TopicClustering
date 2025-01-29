import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from utils import evaluate_clustering, save_metrics, save_cluster_labels, visualize_clusters
import os

def main():
    print("\nStarting LDA-based Topic Clustering...\n")
    
    # ===============================
    # 1. Loading and Preprocessing Data
    # ===============================
    
    # Path to the CSV file
    csv_path = '/content/arxiv-GCN-TopicClustering/data/processed/papers_preprocessed.csv'
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"CSV loaded with shape: {df.shape}")
    
    # Combine title and abstract
    df['text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    
    # Extract texts
    data_texts = df['text'].tolist()
    print(f"Number of documents: {len(data_texts)}")
    
    # Preprocess texts using utility function
    from utils import preprocess_text
    processed_texts = preprocess_text(data_texts)
    print(f"Number of processed texts: {len(processed_texts)}")
    
    # ===============================
    # 2. Creating Dictionary and Corpus
    # ===============================
    
    # Create dictionary
    dictionary = corpora.Dictionary(processed_texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    print(f"Dictionary created with {len(dictionary)} tokens.")
    
    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # ===============================
    # 3. Training LDA Model
    # ===============================
    
    num_topics = 39  # Should match the number of classes
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         random_state=42,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)
    
    # Display topics
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}\n")
    
    # ===============================
    # 4. Assigning Topics to Documents
    # ===============================
    
    # Get topic distributions
    topic_distributions = lda_model.get_document_topics(corpus, minimum_probability=0.0)
    
    # Assign each document to the topic with the highest probability
    lda_cluster_labels = [max(doc, key=lambda x: x[1])[0] for doc in topic_distributions]
    print(f"Sample cluster labels: {lda_cluster_labels[:10]}")
    
    # ===============================
    # 5. Dimensionality Reduction with TruncatedSVD
    # ===============================
    
    # Convert tokens back to strings for vectorization
    texts_joined = [' '.join(text) for text in processed_texts]
    
    # Vectorize texts
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts_joined)
    print(f"Document-term matrix shape: {X.shape}")
    
    # Reduce dimensions
    n_components = 100
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    print(f"Reduced matrix shape: {X_reduced.shape}")
    
    # ===============================
    # 6. Evaluating Clustering Quality
    # ===============================
    
    # Load true labels
    labels_csv_path = "/content/arxiv-GCN-TopicClustering/data/processed/labels_preprocessed.csv"
    df_labels = pd.read_csv(labels_csv_path)
    
    # Load papers to ensure alignment
    papers_csv_path = "/content/arxiv-GCN-TopicClustering/data/processed/papers_preprocessed.csv"
    df_papers = pd.read_csv(papers_csv_path)
    node_order = df_papers['node_id'].tolist()
    
    # Align true labels with cluster labels
    true_labels = df_labels.set_index('node_id').loc[node_order]['label'].tolist()
    
    if len(true_labels) != len(lda_cluster_labels):
        print(f"Mismatch in number of samples: true_labels has {len(true_labels)}, lda_cluster_labels has {len(lda_cluster_labels)}")
        # Truncate to the minimum length
        common_length = min(len(true_labels), len(lda_cluster_labels))
        true_labels = true_labels[:common_length]
        lda_cluster_labels = lda_cluster_labels[:common_length]
    
    # Evaluate clustering
    metrics = evaluate_clustering(true_labels, lda_cluster_labels, X_reduced)
    print("\n--- LDA-based Clustering Evaluation Metrics ---")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: Not computed.")
    
    # Save metrics
    metrics_save_path = "results/metrics/lda_clustering_metrics.pkl"
    save_metrics(metrics, metrics_save_path)
    
    # Save cluster labels
    cluster_labels_save_path = "results/metrics/lda_cluster_labels.pkl"
    save_cluster_labels(lda_cluster_labels, cluster_labels_save_path)
    
    # ===============================
    # 7. Visualize Clustering Results with t-SNE
    # ===============================
    
    print("\nVisualizing LDA-based Clustering Results with t-SNE...")
    
    # Visualize using utility function
    visualization_save_path = "results/plots/lda_tsne.png"
    visualize_clusters(
        embeddings=X_reduced,  # Use reduced dimensions for t-SNE
        cluster_labels=lda_cluster_labels,
        num_clusters=num_topics,
        title='t-SNE Visualization of LDA-based Clustering',
        save_path=visualization_save_path
    )

if __name__ == "__main__":
    main()
