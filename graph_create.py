import networkx as nx
import numpy as np
from sentence_transformers import util
import matplotlib.pyplot as plt



class GraphBuilder:
    def __init__(self, output_dir, topics=None, concept='kmeans'):
        """
        Initialize the GraphBuilder class.

        Parameters:
        - output_dir (str): Directory to save the output GraphML files.
        - topics (list): List of topics.
        """
        self.output_dir = output_dir
        self.topics = topics or ["watching", "episode", "movie", "film", "like", "good", "bad", "arts", "think"]
        self.topic_embeddings = None
        self.concept = concept

    def set_topic_embeddings(self, embeddings):
        """
        Set the topic embeddings.

        Parameters:
        - embeddings (np.ndarray): Topic embeddings.
        """
        self.topic_embeddings = embeddings

    def make_graph(self, doc_embeddings, OOD_embeddings, knn_mode=False, threshold=0.17, k=4, plot_graph=False):
        """
        Create a graph connecting topics, ID documents, and OOD documents.

        Parameters:
        - doc_embeddings (np.ndarray): Embeddings of in-distribution (ID) documents.
        - OOD_embeddings (np.ndarray): Embeddings of out-of-distribution (OOD) documents.
        - knn_mode (bool): Whether to use k-nearest neighbors for edge creation.
        - threshold (float): Similarity threshold for edge creation (if not using KNN).
        - k (int): Number of nearest neighbors (if knn_mode is True).
        - plot_graph (bool): Whether to plot the graph.

        Returns:
        - nx.Graph: The created graph.
        """
        if self.topic_embeddings is None:
            raise ValueError("Topic embeddings must be set before building the graph.")

        G = nx.Graph()

        # Add topics as nodes
        for i, topic_embedding in enumerate(self.topic_embeddings):
            G.add_node(
                self.topics[i],
                type="topic",
                bipartite=0,
                embedding=",".join(map(str, topic_embedding))  # Convert list to string
            )

        if knn_mode:
            # Add edges using k-nearest neighbors
            self._add_knn_edges(G, doc_embeddings, OOD_embeddings, k)
        else:
            # Add edges using a similarity threshold
            self._add_threshold_edges(G, doc_embeddings, OOD_embeddings, threshold)

        # Save the graph
        doc_length = len(doc_embeddings)
        ood_length = len(OOD_embeddings)
        filename = f"{self.concept}_{doc_length}ID_{ood_length}OOD_{knn_mode}_{k}_graph.graphml"
        nx.write_graphml(G, f"{self.output_dir}/{filename}")
        print(f"Graph saved to {self.output_dir}/{filename}")

        # Optionally plot the graph
        if plot_graph:
            self._plot_graph(G, self.output_dir)

        return G

    def _add_knn_edges(self, G, doc_embeddings, OOD_embeddings, k):
        """Add edges using k-nearest neighbors."""
        self._connect_knn_nodes(G, doc_embeddings, k, label=0, prefix="ID_doc")
        self._connect_knn_nodes(G, OOD_embeddings, k, label=1, prefix="OOD_doc")

    def _connect_knn_nodes(self, G, embeddings, k, label, prefix):
        """Connect nodes to topics using k-nearest neighbors."""
        for i, embedding in enumerate(embeddings):
            G.add_node(
                f"{prefix}{i}",
                type=f"{prefix} document",
                bipartite=1,
                label=label,
                embedding=",".join(map(str, embedding))  # Convert list to string
            )
            similarities = [(util.cos_sim(topic_emb, embedding).item(), j) for j, topic_emb in enumerate(self.topic_embeddings)]
            top_k_similarities = sorted(similarities, reverse=True, key=lambda x: x[0])[:k]
            for similarity, j in top_k_similarities:
                G.add_edge(self.topics[j], f"{prefix}{i}", weight=similarity)

    def _add_threshold_edges(self, G, doc_embeddings, OOD_embeddings, threshold):
        """Add edges using a similarity threshold."""
        self._connect_threshold_nodes(G, doc_embeddings, threshold, label=0, prefix="ID_doc")
        self._connect_threshold_nodes(G, OOD_embeddings, threshold, label=1, prefix="OOD_doc")

    def _connect_threshold_nodes(self, G, embeddings, threshold, label, prefix):
        """Connect nodes to topics using a similarity threshold."""
        for i, embedding in enumerate(embeddings):
            G.add_node(
                f"{prefix}{i}",
                type=f"{prefix} document",
                bipartite=1,
                label=label,
                embedding=",".join(map(str, embedding))  # Convert list to string
            )
            for j, topic_emb in enumerate(self.topic_embeddings):
                similarity = util.cos_sim(topic_emb, embedding).item()
                if similarity > threshold:
                    G.add_edge(self.topics[j], f"{prefix}{i}", weight=similarity)

    def _plot_graph(self, G, output_dir):
        """Plot the graph."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=500, font_size=5, font_weight="bold")
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        labels = nx.get_node_attributes(G, "label")
        nx.draw_networkx_labels(G, pos, labels=labels)
        plt.title("Graph of Topics and Documents")
        plt.savefig(output_dir)
