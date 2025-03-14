import os
import pandas as pd
import torch
import time
from itertools import product
from preProcess import preprocess_text
from extract_concepts import ExtractConcepts
from graph_create import GraphBuilder
from model.utils import process_data_from_graph, split_data, initialize_centers_from_topic_nodes, compute_metrics, get_original_data
from model.CoGOOD_model import GCN
from model.training import train
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

def visualize_embeddings_with_labels(embeddings, labels, title, output_dir=None):
    """
    Visualize embeddings using TSNE and PCA with corresponding labels.
    Utilizes GPU for the embeddings processing until visualization.
    """
    # Check if the embeddings and labels are on the GPU, move them to CPU for visualization
    device = embeddings.device if isinstance(embeddings, torch.Tensor) else None
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



# Define output directory
output_dir = "./output_20news"
os.makedirs(output_dir, exist_ok=True)

# Load Data
DATASET_NAME = "20newspaper"
dataset_dir = os.path.join(output_dir, DATASET_NAME)
os.makedirs(dataset_dir, exist_ok=True)

df = pd.read_csv("/home/user01/CoGood/newspaper_all_out.csv")
df['text'] = df['text'].apply(lambda x: str(x) if isinstance(x, str) else str(x) if x is not None else '')
cleaned_df = preprocess_text(df, "text", save_path=os.path.join(dataset_dir, "cleaned_df.csv"))
df = pd.read_csv("/home/user01/CoGood/newspaper_all_out.csv")

# Extract ID and OOD documents
ID_raw_documents = df[df["label"] == "ID"]["text"].tolist()
ID_documents = cleaned_df[cleaned_df["label"] == "ID"]["text"].tolist()
OOD_documents = cleaned_df[cleaned_df["label"] == "OOD"]["text"].tolist()

# Define hyperparameter grid
methods = ["kmeans"]
num_clusters_list = [50]
k_list = [2]
aggr_list = ['max']

results = []

t1 = time.time()
print("Start")
# Extract sentence concepts
# concept_extrcator_sentence = ExtractConcepts(method='sentence')
# all_concepts_sentence, all_concept_embeddings_sentence = concept_extrcator_sentence.extract_concept(ID_raw_documents, OOD_documents, top_n=60)
print("Grid seart Start")
# Perform Grid Search
for method, num_clusters, k, aggr in product(methods, num_clusters_list, k_list, aggr_list):
    exp_dir = os.path.join(output_dir, f'{method}_{num_clusters}_clusters_{k}_knn_{aggr}')
    os.makedirs(exp_dir, exist_ok=True)

    # Extract Concepts
    concept_extractor = ExtractConcepts(method=method)
    if method == "kmeans":
        concept_embeddings = concept_extractor.extract_concept(ID_documents=ID_documents, num_clusters=num_clusters)
        concepts = [f'center{i}' for i in range(num_clusters)]
    else:
        concepts = all_concepts_sentence[:num_clusters]
        concept_embeddings = all_concept_embeddings_sentence[:num_clusters]
        #concepts, concept_embeddings = concept_extractor.extract_concept(ID_raw_documents, OOD_documents, top_n=num_clusters)

    # Build Graph
    graph_builder = GraphBuilder(output_dir=exp_dir, concept=method, topics=concepts)
    graph_builder.set_topic_embeddings(concept_embeddings)
    ID_embeddings = concept_extractor.extract_embeddings(ID_documents)
    OOD_embeddings = concept_extractor.extract_embeddings(OOD_documents)
    G = graph_builder.make_graph(ID_embeddings, OOD_embeddings, knn_mode=True, k=k)

    # Convert Graph to PyTorch Data
    data = process_data_from_graph(G)
    train_data, val_data, test_data, index_mappings = split_data(data)

    # Initialize Model
    model = GCN(out_channels=1, aggr=aggr)
    centers = initialize_centers_from_topic_nodes(G, num_centers=num_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Train Model
    trained_model = train(model, train_data, val_data, optimizer, centers, criterion, max_epochs=5000, output_dir=exp_dir)

    # Evaluate Model
    with torch.no_grad():
        # Move test data to the same device as the model
        device = next(model.parameters()).device  # Get the device where the model is located
        test_data = test_data.to(device)  # Move test data to the same device
        
        y_true = test_data.y.squeeze()
        y_pred_prob, emb = model(test_data)
        
        topic_mask = test_data.y != -1
    
    # Compute metrics on the topic mask
    auroc, aupr, fpr_at_95_tpr = compute_metrics(y_true[topic_mask], y_pred_prob[topic_mask], exp_dir)

    visualize_embeddings_with_labels(
        embeddings=test_data.x,
        labels=test_data.y,
        title="Embeddings before Training",
        output_dir=exp_dir
    )
    visualize_embeddings_with_labels(
        embeddings=emb,
        labels=test_data.y,
        title="Embeddings After Training",
        output_dir=exp_dir
    )

    # Store Results
    results.append({
        "method": method,
        "num_clusters": num_clusters,
        "k": k,
        "aggr": aggr,
        "AUROC": auroc,
        "AUPR": aupr,
        "FPR@95TPR": fpr_at_95_tpr
    })

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(results[-1], f, indent=4)

    # Save Model
    torch.save(trained_model.state_dict(), os.path.join(exp_dir, "trained_model.pth"))

    # Compute ROC curve
    # Move tensors to CPU and convert to numpy before passing them to sklearn functions
    y_true_cpu = y_true[topic_mask].cpu().numpy()  # Move to CPU and convert to numpy
    y_pred_prob_cpu = y_pred_prob[topic_mask].cpu().numpy()  # Move to CPU and convert to numpy
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_cpu, y_pred_prob_cpu)
    
    # Compute Youden's J statistic
    j_scores = tpr - fpr

    # Find the optimal threshold (maximizing Youden's J statistic)
    best_threshold = thresholds[np.argmax(j_scores)]
    y_pred_class = (y_pred_prob > best_threshold) + 0  # Convert probabilities to binary labels

    report = classification_report(y_true[topic_mask].cpu(), y_pred_class[topic_mask].cpu(), output_dict=True)
    # Save the report to a file in JSON format
    with open(os.path.join(exp_dir, f"classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    for class_label in [0, 1]:
        y_pred_class = y_pred_class.view(-1).float()
        wrong_indices = (y_pred_class != y_true).nonzero(as_tuple=True)[0]
        # Get misclassified indices for class 0 and class 1 separately
        wrong_class = wrong_indices[y_true[wrong_indices] == class_label].tolist()
        wrong_class = wrong_class[:5]

        original_info = get_original_data(df, concepts, test_data, index_mappings["test"], wrong_class)
        converted_results = {int(k): v for k, v in original_info.items()}

        with open(os.path.join(exp_dir, f"explain_wrong_predictions_class_{class_label}.json"), "w") as f:
            json.dump(converted_results, f, indent=4)

    for class_label in [0, 1]:
        y_pred_class = y_pred_class.view(-1).float()

        # Get correctly classified indices
        correct_indices = (y_pred_class == y_true).nonzero(as_tuple=True)[0]

        # Filter correctly classified indices for class 0 and class 1
        correct_class = correct_indices[y_true[correct_indices] == class_label].tolist()

        # Select up to 5 samples
        correct_class = correct_class[:5]

        # Get original information for the selected samples
        original_info = get_original_data(df, concepts, test_data, index_mappings["test"], correct_class)
        converted_results = {int(k): v for k, v in original_info.items()}

        with open(os.path.join(exp_dir, f"explain_right_predictions_class_{class_label}.json"), "w") as f:
            json.dump(converted_results, f, indent=4)

t2 = time.time()
print("OVERAL TIME:", t2-t1)
# Save all results
with open(os.path.join(output_dir, "grid_search_results.json"), "w") as f:
    json.dump(results, f, indent=4)