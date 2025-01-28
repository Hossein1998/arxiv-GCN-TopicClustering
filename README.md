# Cora GNN Project

![Cora GNN](https://github.com/yourusername/cora_gnn_project/blob/main/assets/cora_gnn_banner.png)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Setup](#setup)
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

The **Cora GNN Project** leverages Graph Convolutional Networks (GCN) and topic modeling (LDA) to perform node classification and clustering on the Cora dataset. The pipeline includes data preprocessing, embedding generation using Hugging Face's LLaMa models, model training, clustering, evaluation, and visualization.

## Features

- **Data Preprocessing**: Cleans and prepares raw data.
- **Embedding Generation**: Utilizes LLaMa models for generating node embeddings.
- **GCN Training**: Trains a GCN for node classification.
- **Clustering**: Implements K-Means and LDA-based clustering.
- **Evaluation & Visualization**: Assesses and visualizes clustering performance.
- **Resume Capability**: Ensures embedding generation can resume from interruptions.

## Directory Structure


cora_gnn_project/ │ ├── data/ │ ├── raw/ │ │ ├── paper_info.csv │ │ ├── edge_list.csv │ │ └── labels.csv │ └── processed/ │ ├── paper_info_preprocessed.csv │ ├── edges_preprocessed.csv │ ├── labels_preprocessed.csv │ ├── edges_sub.csv │ ├── papers_sub.csv │ ├── labels_sub.csv │ ├── embeddings.pt │ ├── partial_embeddings.pt │ └── data.pt │ ├── notebooks/ │ └── exploration.ipynb │ ├── src/ │ ├── init.py │ ├── data_preprocessing.py │ ├── embedding.py │ ├── gnn_training.py │ ├── clustering.py │ ├── lda_clustering.py │ ├── result_analysis.py │ ├── visualization.py │ └── utils.py │ ├── models/ │ ├── gcn_model.pth │ └── node2vec_embeddings.pt │ ├── results/ │ ├── plots/ │ │ ├── gcn_tsne.png │ │ ├── clustering_comparison.png │ │ ├── kmeans_tsne.png │ │ └── lda_tsne.png │ └── metrics/ │ ├── clustering_metrics.pkl │ ├── cluster_labels.pkl │ ├── lda_clustering_metrics.pkl │ └── lda_cluster_labels.pkl │ ├── README.md ├── requirements.txt └── setup.py
