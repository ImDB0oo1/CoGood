import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import os

from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

import networkx as nx
from torch_geometric.data import Data

def initialize_centers(data, model, num_centers=60):
    """
    Initialize multiple centers by clustering the embeddings of the in-distribution data.
    """
    with torch.no_grad():
        model.eval()
        label, embeddings = model(data)  # Get embeddings for training nodes
        embeddings = embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=num_centers, random_state=42).fit(embeddings)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=data.x.device)
    return centers

def initialize_centers_from_topic_nodes(G, num_centers=60):
    """
    Initialize centers from the topic nodes' embeddings.
    """
    topic_embeddings = []
    for node, data in G.nodes(data=True):
        if data.get('type') == 'topic':  # Check for topic node type
            topic_embeddings.append(data['embedding'])

    centers = torch.tensor(topic_embeddings, dtype=torch.float)
    centers = centers[:num_centers]  # Take up to `num_centers` topic embeddings
    return centers

def cosine_similarity_distance(embeddings, centers):
    """
    Compute the cosine similarity-based distance to centers.
    Negative cosine similarity is treated as a 'distance.'
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    centers = F.normalize(centers, p=2, dim=1)
    similarity = torch.matmul(embeddings, centers.T)
    distance = 1 - similarity
    return distance

def assign_to_centers(embeddings, centers):
    """
    Assign each embedding to the nearest center based on cosine similarity.
    """
    distances = cosine_similarity_distance(embeddings, centers)
    min_distances, assigned_centers = torch.min(distances, dim=1)
    return min_distances, assigned_centers

def multi_center_svdd_loss(embeddings, centers):
    """
    Compute the multi-center Deep SVDD loss using cosine similarity distance.
    """
    # Calculate cosine similarity-based distances to all centers
    distances = cosine_similarity_distance(embeddings, centers)  # Shape: [num_nodes, num_centers]

    # Find the minimum distance for each embedding
    min_distances, _ = torch.min(distances, dim=1)  # Shape: [num_nodes]

    # Return the mean of the minimum distances
    return torch.mean(min_distances ** 2)

def compute_metrics(y_true, y_pred_prob, exp_dir=None):
    """
    Compute and plot AUROC, AUPR, and FPR@95TPR, and save the plots in exp_dir.
    """
    y_true = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_prob = y_pred_prob.detach().cpu().numpy() if isinstance(y_pred_prob, torch.Tensor) else y_pred_prob
    
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
    auroc = auc(fpr, tpr)
    
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(recall, precision)
    
    threshold_95_tpr = thresholds_roc[np.argmax(tpr >= 0.95)]
    fpr_at_95_tpr = fpr[np.argmax(tpr >= 0.95)]
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {auroc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(exp_dir, "roc_curve.png"))
    plt.close()
    
    plt.figure()
    plt.plot(recall, precision, label=f"Precision-Recall Curve (AUPR = {aupr:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(exp_dir, "precision_recall_curve.png"))
    plt.close()
    
    plt.figure()
    plt.plot(thresholds_roc, fpr, label="FPR vs Threshold")
    plt.axvline(threshold_95_tpr, color='red', linestyle='--', label=f"Threshold @ 95% TPR = {threshold_95_tpr:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("False Positive Rate")
    plt.title("FPR at Various Thresholds")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(exp_dir, "fpr_threshold_curve.png"))
    plt.close()
    
    return auroc, aupr, fpr_at_95_tpr

def process_data_from_graph(G):
    """
    Process graph data and create node features, labels, masks, and other necessary attributes.

    Parameters:
    - G: NetworkX graph with node features and labels.
    - num_samples_ID: Number of ID (In-Distribution) nodes to sample.
    - num_samples_OOD: Number of OOD (Out-Of-Distribution) nodes to sample.

    Returns:
    - data: PyTorch Geometric Data object.
    - node_features: Node features.
    - node_labels: Node labels.
    - train_mask: Mask for training nodes.
    - node_mask: Mask for all non-topic nodes.
    - isolated_mask: Mask for isolated nodes.
    """
    for node_id, node_data in G.nodes(data=True):
      if 'embedding' in node_data:
          # Convert string back to numpy array
          #print(node_data['type'], ":", len(node_data['embedding']))
          node_data['embedding'] = np.fromstring(node_data['embedding'], sep=',')

    # Assuming that 'embedding' and 'label' are attributes stored in the nodes (adjust if different)
    for node, data in G.nodes(data=True):
        if 'embedding' not in data:
            data['embedding'] = np.random.rand(384).tolist()  # Random embeddings if not present
        if 'label' not in data:
            data['label'] = -1  # Random labels if not present

    node_features = []
    node_labels = []
    for node, data in G.nodes(data=True):
        node_features.append(data['embedding'] + np.random.normal(0, 0, data['embedding'].shape))  # Assuming 'embedding' is already a list of floats
        node_labels.append(data['label'])  # Node labels (ID or OOD)

    # Convert data to PyTorch tensors
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    y = torch.tensor(node_labels, dtype=torch.float)

    # Create the PyTorch Geometric data object
    data = Data(x=x, y=y)

    # Create edge data (edge indices and weights)

    edge_index, edge_weight = create_edge_data(G)

    # Add the edge data to the PyTorch Geometric data object
    data.edge_index = edge_index
    data.edge_weight = edge_weight

    return data


def create_edge_data(G):
    """
    Convert graph edges to edge indices and weights for PyTorch Geometric format.

    Parameters:
    - G: NetworkX graph.
    - subgraph_nodes: Nodes for which edges should be included (if provided).

    Returns:
    - edge_index: Edge indices in PyTorch tensor format.
    - edge_weight: Edge weights as a PyTorch tensor.
    """
    # Preserve node order
    unique_nodes = list(G.nodes())  # Get nodes in the original order
    node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}  # Map nodes to indices

    edge_indices = [[node_to_index[src], node_to_index[dst]] for src, dst in G.edges()]
    edge_weights = [G[src][dst].get('weight', 1.0) for src, dst in G.edges()]  # Default weight is 1.0 if not provided

    # Convert to PyTorch tensors
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    return edge_index, edge_weight

def split_data(data, train_ratio=0.6, val_ratio=0.2):
    """
    Splits the PyTorch Geometric Data object into train, validation, and test sets,
    ensuring topic nodes are included in all splits while masking both nodes and edges.
    Also maintains a mapping to revert indices back to the original dataset.

    Parameters:
    - data: PyTorch Geometric Data object.
    - train_ratio: Fraction of data to use for training.
    - val_ratio: Fraction of data to use for validation.

    Returns:
    - train_data, val_data, test_data: PyTorch Geometric Data objects.
    - index_mappings: Dictionary mapping new indices to original indices for each split.
    """
    num_nodes = data.x.shape[0]
    topic_mask = (data.y == -1)  # Assuming topic nodes have label -1
    non_topic_indices = np.where(~topic_mask.numpy())[0]
    np.random.shuffle(non_topic_indices)

    train_size = int(train_ratio * len(non_topic_indices))
    val_size = int(val_ratio * len(non_topic_indices))

    train_indices = non_topic_indices[:train_size]
    val_indices = non_topic_indices[train_size:train_size + val_size]
    test_indices = non_topic_indices[train_size + val_size:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Ensure topic nodes are included in all splits
    train_mask[topic_mask] = True
    val_mask[topic_mask] = True
    test_mask[topic_mask] = True

    # Get valid node indices for each split
    train_nodes = torch.where(train_mask)[0]
    val_nodes = torch.where(val_mask)[0]
    test_nodes = torch.where(test_mask)[0]

    # Create mappings from new indices to original indices
    train_index_map = {i: idx.item() for i, idx in enumerate(train_nodes)}
    val_index_map = {i: idx.item() for i, idx in enumerate(val_nodes)}
    test_index_map = {i: idx.item() for i, idx in enumerate(test_nodes)}

    # Create a mapping from old indices to new indices
    node_mapping = -torch.ones(num_nodes, dtype=torch.long)
    node_mapping[train_nodes] = torch.arange(train_nodes.size(0))
    node_mapping[val_nodes] = torch.arange(val_nodes.size(0))
    node_mapping[test_nodes] = torch.arange(test_nodes.size(0))

    # Mask edges and remap node indices
    edge_index = data.edge_index
    edge_mask_train = train_mask[edge_index[0]] & train_mask[edge_index[1]]
    edge_mask_val = val_mask[edge_index[0]] & val_mask[edge_index[1]]
    edge_mask_test = test_mask[edge_index[0]] & test_mask[edge_index[1]]

    train_edge_index = node_mapping[edge_index[:, edge_mask_train]]
    val_edge_index = node_mapping[edge_index[:, edge_mask_val]]
    test_edge_index = node_mapping[edge_index[:, edge_mask_test]]

    edge_weight = data.edge_weight
    train_edge_weight = edge_weight[edge_mask_train] if edge_weight is not None else None
    val_edge_weight = edge_weight[edge_mask_val] if edge_weight is not None else None
    test_edge_weight = edge_weight[edge_mask_test] if edge_weight is not None else None

    train_data = Data(
        x=data.x[train_mask], y=data.y[train_mask], edge_index=train_edge_index,
        edge_weight=train_edge_weight, train_mask=train_mask
    )
    val_data = Data(
        x=data.x[val_mask], y=data.y[val_mask], edge_index=val_edge_index,
        edge_weight=val_edge_weight, val_mask=val_mask
    )
    test_data = Data(
        x=data.x[test_mask], y=data.y[test_mask], edge_index=test_edge_index,
        edge_weight=test_edge_weight, test_mask=test_mask
    )

    index_mappings = {
        "train": train_index_map,
        "val": val_index_map,
        "test": test_index_map
    }

    return train_data, val_data, test_data, index_mappings

def get_original_data(df, node_list, split_data, index_mapping, node_indices):
    """
    Given a list of node indices in a split dataset, return their original indices,
    their values from the dataframe, the nodes they connect to, and edge weights.

    Parameters:
    - df: Original DataFrame where indices match the original dataset.
    - node_list: List where indices correspond to actual node indices in the dataset.
    - split_data: The split PyTorch Geometric Data object (train, val, or test).
    - index_mapping: Mapping from split dataset indices to original dataset indices.
    - node_indices: List of node indices in the split dataset.

    Returns:
    - results: Dictionary containing original indices, values from df, connected nodes, and edge weights.
    """
    original_indices = np.array([index_mapping.get(idx, None) for idx in node_indices])
    valid_mask = original_indices != None
    original_indices = original_indices[valid_mask]
    node_indices = np.array(node_indices)[valid_mask]

    df_values = df.loc[original_indices] if len(original_indices) > 0 else None

    # Initialize dictionary to store results
    connections_dict = {}

    # Loop through each node and find its connected source nodes
    for node in node_indices:
        mask = split_data.edge_index[1] == node  # Find edges where node is the target
        connected_nodes = split_data.edge_index[0, mask]  # Get corresponding source nodes
        connected_weights = split_data.edge_weight[mask]
        connections_dict[int(node)] = {
            node_list[i]: connected_weights[j].item() for j, i in enumerate(torch.unique(connected_nodes))
        }

    results = {
        idx: {
            "df_value": df_values.loc[idx][0] if df_values is not None else None,
            "connected_nodes": connections_dict.get(int(node_indices[i]), []),
        }
        for i, idx in enumerate(original_indices)
    }

    return results

import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embeddings_with_labels(embeddings, labels, title, output_dir=None):
    """
    Visualize embeddings using TSNE and PCA with corresponding labels.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()

    # TSNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = labels == label
        plt.scatter(
            tsne_result[indices, 0],
            tsne_result[indices, 1],
            s=10,
            alpha=0.7,
            label=f"Label {int(label)}"
        )
    plt.title(f"{title} - TSNE")
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend()
    plt.grid(True)

    if output_dir:
        tsne_path = os.path.join(output_dir, f"{title}_TSNE.png")
        plt.savefig(tsne_path, dpi=300)
        print(f"TSNE visualization saved to {tsne_path}")
    plt.close()

    # PCA Visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        indices = labels == label
        plt.scatter(
            pca_result[indices, 0],
            pca_result[indices, 1],
            s=10,
            alpha=0.7,
            label=f"Label {int(label)}"
        )
    plt.title(f"{title} - PCA")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)

    if output_dir:
        pca_path = os.path.join(output_dir, f"{title}_PCA.png")
        plt.savefig(pca_path, dpi=300)
        print(f"PCA visualization saved to {pca_path}")
    plt.close()
