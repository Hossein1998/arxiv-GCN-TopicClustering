![image](https://github.com/user-attachments/assets/b8f339f3-d26b-449f-ac04-377db38596a1)## Table of Contents

- [Overview](#overview)
- [Features](#features)
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

The **arxiv GNN Project** leverages Graph Convolutional Networks (GCN) and topic modeling (LDA) to perform node classification and clustering on the arxiv dataset. The pipeline includes data preprocessing, embedding generation using Hugging Face's LLaMa models, model training, clustering, evaluation, and visualization.

## Features

- **Data Preprocessing**: Cleans and prepares raw data.
- **Embedding Generation**: Utilizes LLaMa models for generating node embeddings.
- **GCN Training**: Trains a GCN for node classification.
- **Clustering**: Implements K-Means and LDA-based clustering.
- **Evaluation & Visualization**: Assesses and visualizes clustering performance.
- **Resume Capability**: Ensures embedding generation can resume from interruptions.

