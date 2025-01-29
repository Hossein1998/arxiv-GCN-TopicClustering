## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Embedding Generation](#2-embedding-generation)
  - [3. GCN Training](#3-gcn-training)
  - [4. K-Means Clustering](#4-k-means-clustering)
  - [5. LDA-based Clustering](#5-lda-based-clustering)
  - [6. Result Analysis](#6-result-analysis)
  - [7. Visualization](#7-visualization)
- [Running on Google Colab](#running-on-google-colab)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The **arxiv GNN Project** leverages Graph Convolutional Networks (GCN) and topic modeling (LDA) to perform node classification and clustering on the arxiv dataset. The pipeline includes data preprocessing, embedding generation using Hugging Face's LLaMa models, model training, clustering, evaluation, and visualization.

## Features

- **Data Preprocessing**: Cleans and prepares raw data.
- **Embedding Generation**: Utilizes LLaMa models for generating node embeddings.
- **GCN Training**: Trains a GCN for node classification.
- **Clustering**: Implements K-Means and LDA-based clustering.
- **Evaluation & Visualization**: Assesses and visualizes clustering performance.
- **Resume Capability**: Ensures embedding generation can resume from interruptions.

## Installation

To set up the project and install the required dependencies, follow these steps:

### 1. Clone the Repository

Clone the project repository using the following command:

```bash
git clone https://github.com/Hossein1998/arxiv-GCN-TopicClustering.git
cd arxiv-GCN-TopicClustering
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```
## Usage

### 1. Data Preprocessing
- Load and clean the abstract and title of each paper (e.g., CSV format).
- Remove missing values from key columns (`title`, `abstract`).
- Perform text cleaning (lowercasing, removing punctuation, etc.).
- Filter and subset data based on valid node IDs.
- Save the preprocessed data for further use.

Run:
```bash
python preprocess_data.py
```

### 2. Embedding Generation

- Load a pretrained model (e.g., LLaMA).
- Generate **title and abstract embeddings** for each paper.
- Save embeddings and processed graph data for future use.

Run:
```bash
python generate_embeddings.py
```

### 3. GCN Training

- Load **title and abstract embeddings** along with graph data.
- Train a **Graph Convolutional Network (GCN)** for node classification.
- Use **early stopping** and validation accuracy to select the best model.
- Extract hidden embeddings for clustering.
- Save the trained model and embeddings for downstream tasks.

Run:
```bash
python gnn_training.py
```

### 4. K-Means Clustering

- Apply **K-Means clustering** on **GCN hidden embeddings**.
- Evaluate clustering quality using various metrics.
- Save cluster labels and performance metrics.
- Visualize clustering results with **t-SNE**.

Run:
```bash
python kmeans_clustering.py
```

### 5. LDA-based Clustering

- Apply **Latent Dirichlet Allocation (LDA)** to cluster papers based on **title and abstract**.
- Reduce dimensionality using **TruncatedSVD** for better clustering performance.
- Assign topics to each document based on the **highest probability**.
- Evaluate clustering results and visualize topic distributions.

Run:
```bash
python lda_clustering.py
```

### 6. Result Analysis

- Evaluate clustering results using various metrics:
  - **Silhouette Score**
  - **Adjusted Rand Index (ARI)**
  - **Normalized Mutual Information (NMI)**
- Compare **K-Means** and **LDA-based clustering** performance.

Run:
```bash
python result_analysis.py
```


### 7. Visualization
- Convert the **graph structure** into a **NetworkX** visualization.
- Reduce dimensionality using **t-SNE** for better cluster visualization.
- Save high-quality plots for analysis.

Run:
```bash
python visualize_clusters.py
```
