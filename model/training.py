import torch
from model.utils import multi_center_svdd_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Ensure CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_data, val_data, optimizer, centers, criterion, max_epochs=100, patience=10, output_dir=None):
    # Move the model to GPU
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    train_losses = []
    val_losses = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        
        # Move data to GPU
        train_data = train_data.to(device)
        
        optimizer.zero_grad()
        label, embeddings = model(train_data)
        
        topic_mask = train_data.y != -1
        #svdd_loss = multi_center_svdd_loss(embeddings, centers)
        label_loss = criterion(label[topic_mask].squeeze(), train_data.y[topic_mask])
        
        #svdd_weight = min(1.0, epoch / 10)  # Gradual increase over 10 epochs
        label_weight = 0.5
        #loss = svdd_weight * svdd_loss + label_weight * label_loss   
        loss = label_weight * label_loss        
     
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss: {loss.item()}")        
       
        val_loss = evaluate(model, val_data, criterion, centers, epoch)
        print(f"Epoch {epoch+1}: val Loss: {val_loss}")        
        train_losses.append(loss.item())
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = model.state_dict()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
    
    model.load_state_dict(best_model)
    
    # Plot Training & Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss', color='blue')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss', color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    return model

def evaluate(model, data, criterion, centers, epoch):
    # Move the data to GPU
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        label, embeddings = model(data)
        topic_mask = data.y != -1
        #svdd_loss = multi_center_svdd_loss(embeddings, centers)
        label_loss = criterion(label[topic_mask].squeeze(), data.y[topic_mask])

        #svdd_weight = min(1.0, epoch / 10)  # Same gradual increase as in training
        label_weight = 0.5
        #loss = svdd_weight * svdd_loss + label_weight * label_loss
        loss =  label_weight * label_loss

    return loss.item()

