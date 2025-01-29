import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import os

def load_data(file_path: str) -> Data:
    """
    Load the PyTorch Geometric Data object from a .pt file.

    Args:
        file_path (str): Path to the .pt file.

    Returns:
        Data: Loaded PyTorch Geometric Data object.
    """
    try:
        data = torch.load(file_path)
        print(f"Data loaded successfully from {file_path}")
        print(data)
        return data
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def data_to_networkx(data: Data) -> nx.Graph:
    """
    Convert a PyTorch Geometric Data object to a NetworkX Graph.

    Args:
        data (Data): PyTorch Geometric Data object.

    Returns:
        nx.Graph: Converted NetworkX graph.
    """
    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes with labels
    num_nodes = data.num_nodes
    print(f"Number of nodes: {num_nodes}")
    G.add_nodes_from(range(num_nodes))  # Nodes are labeled from 0 to num_nodes-1

    # Add edges
    edge_index = data.edge_index.numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    print(f"Number of edges: {len(edges)}")
    G.add_edges_from(edges)

    return G

def visualize_graph(G: nx.Graph, subset: bool = True, subset_size: int = 1000):
    """
    Visualize the graph. For large graphs, visualize a subset.

    Args:
        G (nx.Graph): NetworkX graph.
        subset (bool): Whether to visualize a subset of the graph.
        subset_size (int): Number of nodes to include in the subset.
    """
    plt.figure(figsize=(20, 20))

    if subset and G.number_of_nodes() > subset_size:
        # Extract the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc).copy()
        print(f"Largest connected component has {subgraph.number_of_nodes()} nodes.")
        
        # If the largest component is larger than subset_size, sample
        if subgraph.number_of_nodes() > subset_size:
            sampled_nodes = random.sample(largest_cc, subset_size)
            subgraph = G.subgraph(sampled_nodes).copy()
            print(f"Sampled {subset_size} nodes from the largest connected component.")
        
        G_to_draw = subgraph
    else:
        G_to_draw = G

    # Use a spring layout; for very large graphs, consider using other layouts or visualizing a subset
    try:
        pos = nx.spring_layout(G_to_draw, seed=42, k=0.1)  # k controls the distance between nodes
        plt.figure(figsize=(20, 20))
        nx.draw(
            G_to_draw, pos,
            node_size=10,
            node_color='blue',
            edge_color='gray',
            alpha=0.5
        )
        plt.title("Graph Visualization")
        plt.axis('off')
        
        # Create the directory if it doesn't exist
        os.makedirs("results/plots", exist_ok=True)
        plot_save_path = "results/plots/network_visualization.png"
        plt.savefig(plot_save_path, dpi=300)
        plt.show()
        print(f"Graph visualization saved to {plot_save_path}")
    except Exception as e:
        print(f"An error occurred during graph visualization: {e}")
        print("Consider visualizing a smaller subset of the graph.")

def main():
    # Path to the saved PyTorch Geometric Data object
    data_pt_path = 'data/processed/data.pt'

    # Step 1: Load the Data object
    data = load_data(data_pt_path)
    if data is None:
        return

    # Step 2: Convert to NetworkX graph
    G = data_to_networkx(data)

    # Step 3: Visualize the graph
    visualize_graph(G, subset=True, subset_size=1000)  # Adjust subset_size as needed

if __name__ == "__main__":
        main()
