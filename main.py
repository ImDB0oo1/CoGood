import os
import pandas as pd
import torch
from preProcess import preprocess_text
from extract_concepts import ExtractConcepts
from graph_create import GraphBuilder
from model.utils import process_data_from_graph, split_data, initialize_centers, initialize_centers_from_topic_nodes, compute_metrics, get_original_data
from model.CoGOOD_model import GCN,MLP
from model.training import train
import json
import matplotlib.pyplot as plt

import numpy as np
from torch_geometric.data import Data

import pandas as pd
# Define output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Preprocess Data
DATASET_NAME = "IMDB"
dataset_dir = os.path.join(output_dir, DATASET_NAME)
os.makedirs(dataset_dir, exist_ok=True)

ID_df_path = "/content/drive/MyDrive/Thesis_datas/IMDB.csv"
ID_df = pd.read_csv(ID_df_path)  # Update with actual path
ID_df = preprocess_text(ID_df, "text", save_path=os.path.join(dataset_dir, "cleaned_ID.csv"))

OOD_df_path = "/content/drive/MyDrive/Thesis_datas/IMDB.csv"
OOD_df = pd.read_csv(OOD_df_path)  # Update with actual path
OOD_df = preprocess_text(OOD_df, "text", save_path=os.path.join(dataset_dir, "cleaned_OOD.csv"))

ID_documents = ID_df['text'].tolist()
OOD_documents = OOD_df['text'].tolist()

# Step 2: Extract Concepts
method = "kmeans"
if method == "kmeans":
    dataset_dir = os.path.join(output_dir, method)
    os.makedirs(dataset_dir, exist_ok=True)
    concept_extractor = ExtractConcepts(method="kmeans")
    ID_embeddings = concept_extractor.extract_embeddings(ID_documents)
    OOD_embeddings = concept_extractor.extract_embeddings(OOD_documents)
    num_clusters_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for num_clusters in num_clusters_list:
        dataset_dir = os.path.join(output_dir, f'{num_clusters}_concepts')
        os.makedirs(dataset_dir, exist_ok=True)                    
        concept_embeddings = concept_extractor.extract_concept(ID_documents=ID_documents, num_clusters=num_clusters)
        concepts = [f'center{i}' for i in range(num_clusters)]

if method == "sentence":
    dataset_dir = os.path.join(output_dir, method)
    os.makedirs(dataset_dir, exist_ok=True)    
    concept_extractor = ExtractConcepts(method="sentence")
    top_n_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for top_n in top_n_list:
        dataset_dir = os.path.join(output_dir, f'{top_n}_concepts')
        os.makedirs(dataset_dir, exist_ok=True)            
        concepts, concept_embeddings = concept_extractor.extract_concept(ID_documents=ID_documents, OOD_documents=OOD_documents, top_n=top_n)

# Step 3: Build Graph
k_list = [2, 3, 4, 5, 6, 7 ,8]
for k in k_list:
    graph_builder = GraphBuilder(output_dir=dataset_dir, topics=concepts)
    graph_builder.set_topic_embeddings(concept_embeddings)
    G = graph_builder.make_graph(ID_embeddings, OOD_embeddings, knn_mode=True, k=3)

# Step 4: Convert Graph to PyTorch Geometric Data
data = process_data_from_graph(G)
train_data, val_data, test_data, index_mappings = split_data(data)

#train_data, val_data, test_data = split_data(data)

# Step 5: Initialize Model and Centers
model = GCN(out_channels=1)
#centers = initialize_centers(train_data, model)
centers = initialize_centers_from_topic_nodes(G, num_centers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()



# Step 6: Train Model
trained_model = train(model, train_data, val_data, optimizer, centers, criterion, max_epochs=50)

node_indices = [5, 6, 9]  # Example indices from test_data
original_info = get_original_data(df, concepts, test_data, index_mappings["test"], node_indices)
converted_results = {int(k): v for k, v in original_info.items()}
with open("results.json", "w") as f:
    json.dump(converted_results, f, indent=4)

# 8. Evaluate the model and compute metrics
with torch.no_grad():
    y_true = test_data.y.squeeze()  # True labels for non-topic nodes
    y_pred_prob, emb = model(test_data)
    topic_mask = test_data.y != -1  # Only keep non-topic nodes
    y_pred_prob = y_pred_prob.squeeze()  # Model output probabilities

# Compute AUROC, AUPR, and FPR@95TPR
auroc, aupr, fpr_at_95_tpr = compute_metrics(y_true[topic_mask], y_pred_prob[topic_mask])
print(f"AUROC: {auroc:.4f}")
print(f"AUPR: {aupr:.4f}")
print(f"FPR@95TPR: {fpr_at_95_tpr:.4f}")

# 9. Visualize embeddings with labels
# visualize_embeddings_with_labels(
#     embeddings=data.x,
#     labels=data.y,
#     title="Embeddings before Training"
# )
# visualize_embeddings_with_labels(
#     embeddings=emb,
#     labels=data.y,
#     title="Embeddings After Training"
# )
# Compute Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_true[topic_mask], y_pred_prob[topic_mask])

# Compute F1-score for each threshold
f1_scores = (2 * precisions * recalls) / (precisions + recalls)
best_threshold = thresholds[np.argmax(f1_scores[:-1])]  # Ignore last threshold
# 10. Generate classification report and confusion matrix
y_pred_class = (y_pred_prob > best_threshold) + 0  # Convert probabilities to binary labels
print("Classification Report:")
print(classification_report(y_true[topic_mask].cpu(), y_pred_class[topic_mask].cpu()))

# Step 7: Save Model
torch.save(trained_model.state_dict(), os.path.join(dataset_dir, "trained_model.pth"))