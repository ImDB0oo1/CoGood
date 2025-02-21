import os
import pandas as pd
import torch
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

# Define output directory
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Load Data
DATASET_NAME = "IMDB"
dataset_dir = os.path.join(output_dir, DATASET_NAME)
os.makedirs(dataset_dir, exist_ok=True)

ID_df_path = "/content/drive/MyDrive/Thesis_datas/IMDB.csv"
ID_df = pd.read_csv(ID_df_path)
ID_df = preprocess_text(ID_df, "text", save_path=os.path.join(dataset_dir, "cleaned_ID.csv"))
ID_documents = ID_df['text'].tolist()

OOD_df_path = "/content/drive/MyDrive/Thesis_datas/IMDB.csv"
OOD_df = pd.read_csv(OOD_df_path)
OOD_df = preprocess_text(OOD_df, "text", save_path=os.path.join(dataset_dir, "cleaned_OOD.csv"))
OOD_documents = OOD_df['text'].tolist()

# Define hyperparameter grid
methods = ["kmeans", "sentence"]
num_clusters_list = [10, 20, 30, 40, 50]
k_list = [2, 3, 4, 5]
aggr_list = ['max', 'sum', 'mean']

results = []

# Perform Grid Search
for method, num_clusters, k, aggr in product(methods, num_clusters_list, k_list, aggr_list):
    exp_dir = os.path.join(output_dir, f'{method}_{num_clusters}_clusters_{k}_knn_{aggr}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Extract Concepts
    concept_extractor = ExtractConcepts(method=method)
    if method == "kmeans":
        concept_embeddings = concept_extractor.extract_concept(ID_documents, num_clusters)
        concepts = [f'center{i}' for i in range(num_clusters)]
    else:
        concepts, concept_embeddings = concept_extractor.extract_concept(ID_documents, OOD_documents, top_n=num_clusters)
    
    # Build Graph
    graph_builder = GraphBuilder(output_dir=exp_dir, topics=concepts)
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
    trained_model = train(model, train_data, val_data, optimizer, centers, criterion, max_epochs=50)
    
    # Evaluate Model
    with torch.no_grad():
        y_true = test_data.y.squeeze()
        y_pred_prob, emb = model(test_data)
        topic_mask = test_data.y != -1
        auroc, aupr, fpr_at_95_tpr = compute_metrics(y_true[topic_mask], y_pred_prob[topic_mask], exp_dir)
    
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

    # Save original data mapping
    sorted_indices = torch.argsort(y_pred_prob, descending=True)
    class_0_mask = (y_true == 0)
    class_0_indices = sorted_indices[class_0_mask[sorted_indices]]
    correct_indices = class_0_indices[(y_true[class_0_indices] == (y_pred_prob[class_0_indices] > 0.5).int())][:5].tolist()
    incorrect_indices = class_0_indices[(y_true[class_0_indices] != (y_pred_prob[class_0_indices] > 0.5).int())][:5].tolist()
    node_indices = correct_indices + incorrect_indices
    
    original_info = get_original_data(ID_df, concepts, test_data, index_mappings["test"], node_indices)
    converted_results = {int(k): v for k, v in original_info.items()}
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(converted_results, f, indent=4)

# Save all results
with open(os.path.join(output_dir, "grid_search_results.json"), "w") as f:
    json.dump(results, f, indent=4)