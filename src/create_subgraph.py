import gdown
import pandas as pd
import networkx as nx
import random
import os

# ------------------------------
# 1) Download file from Google Drive
# ------------------------------
file_id = '1qhsP1zqjQ2IAwKKzdkpcJ4T1y-2SGrUJ'
url = f"https://drive.google.com/uc?id={file_id}"
output = '/content/arxiv-GCN-TopicClustering/data/raw/paper_info.csv'

print("Downloading file from Google Drive...")
gdown.download(url, output, quiet=False)

# ------------------------------
# 2) Load datasets from specified paths
# ------------------------------
print("Loading datasets...")

# Load datasets from specified paths
edges_df = pd.read_csv('/content/arxiv-GCN-TopicClustering/data/raw/edge_list.csv')
labels_df = pd.read_csv('/content/arxiv-GCN-TopicClustering/data/raw/labels.csv')
papers_df = pd.read_csv(output)

# ------------------------------
# 3) Ensure data consistency by converting node ids to strings
# ------------------------------
print("Ensuring data consistency...")
edges_df[['src', 'dst']] = edges_df[['src', 'dst']].astype(str)
papers_df['node_id'] = papers_df['node_id'].astype(str)
labels_df['node_id'] = labels_df['node_id'].astype(str)

# ------------------------------
# 4) Build graph using NetworkX
# ------------------------------
print("Building graph...")
G = nx.from_pandas_edgelist(edges_df, source='src', target='dst', create_using=nx.Graph())

# ------------------------------
# 5) Find the largest connected component
# ------------------------------
print("Finding largest connected component...")
largest_component = max(nx.connected_components(G), key=len)

# ------------------------------
# 6) Sample nodes from the largest connected component
# ------------------------------
desired_size = 57471
largest_component = list(largest_component)

if len(largest_component) > desired_size:
    sampled_nodes = random.sample(largest_component, desired_size)
    print(f"Sampled {desired_size} nodes from the largest component.")
else:
    sampled_nodes = largest_component
    print(f"Using all {len(largest_component)} nodes.")

sampled_nodes_set = set(sampled_nodes)

# ------------------------------
# 7) Filter datasets based on sampled nodes
# ------------------------------
print("Filtering datasets based on sampled nodes...")

edges_sub_df = edges_df[(edges_df['src'].isin(sampled_nodes_set)) & (edges_df['dst'].isin(sampled_nodes_set))]
papers_sub_df = papers_df[papers_df['node_id'].isin(sampled_nodes_set)]
labels_sub_df = labels_df[labels_df['node_id'].isin(sampled_nodes_set)]

# ------------------------------
# 8) Validate the consistency of sampled data
# ------------------------------
assert papers_sub_df['node_id'].nunique() == len(sampled_nodes_set), "Mismatch in paper nodes!"
assert labels_sub_df['node_id'].nunique() == len(sampled_nodes_set), "Mismatch in label nodes!"

print("Validation successful!")

# ------------------------------
# 9) Check if 'processed' directory exists under the correct path, create if not
# ------------------------------
processed_dir = '/content/arxiv-GCN-TopicClustering/data/processed'
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    print(f"Created directory: {processed_dir}")
else:
    print(f"Directory already exists: {processed_dir}")

# ------------------------------
# 10) Save the filtered datasets in the 'processed' directory
# ------------------------------
print("Saving subset datasets to the 'processed' folder...")

edges_sub_df.to_csv(f"{processed_dir}/edges_sub.csv", index=False)
papers_sub_df.to_csv(f"{processed_dir}/papers_sub.csv", index=False)
labels_sub_df.to_csv(f"{processed_dir}/labels_sub.csv", index=False)

print("Processing complete!")
