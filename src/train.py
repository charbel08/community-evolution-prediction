import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.models import CommunityEvolutionLSTM

# Device configuration: use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def compute_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_t, in loader:
            batch_X = batch_X.to(device)
            batch_t = batch_t.to(device)

            logits = model(batch_X)           # (B, T, 2)
            preds = logits.argmax(dim=-1)     # (B, T)

            mask = batch_t != -1
            correct += (preds[mask] == batch_t[mask]).sum().item()
            total += mask.sum().item()

    return (correct / total) if total > 0 else 0.0


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, run, experiment_name):

    # Make sure model expects masks if you want to use them in forward later
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_loss_history = []
    train_acc_history = []
    val_acc_history = []

    # if experiment == "baseline":
    #     selected_idx = slice(0, input_dim // 2)
    # elif experiment == "topological":
    #     selected_idx = slice(-input_dim // 2, None)
    # else:
    selected_idx = slice(None)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_t in train_loader:
            batch_X = batch_X[:, :, selected_idx].to(device)
            batch_t = batch_t.to(device).long()
            # if experiment == "topological":
            #     mask = torch.all(batch_X == 0, dim=-1)  # True where all 11 features are 0
            #     batch_t[mask] = -1
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits.view(-1, 2), batch_t.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Normalize loss over full dataset
        epoch_loss /= len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        # Accuracy
        train_acc = compute_accuracy(model, train_loader)
        val_acc = compute_accuracy(model, val_loader)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
    print(f"RUN {run} DONE\n")

    return train_loss_history, train_acc_history, val_acc_history


def train(X, t, run, experiment_name):

    # Split dataset into 80-10-10
    train_X, temp_X, train_t, temp_t = train_test_split(
        X, t, test_size=0.2, random_state=42
    )
    val_X, test_X, val_t, test_t = train_test_split(
        temp_X, temp_t, test_size=0.5, random_state=42
    )

    train_dataset = TensorDataset(train_X, train_t)
    val_dataset = TensorDataset(val_X, val_t)
    test_dataset = TensorDataset(test_X, test_t)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Hyperparameters.
    input_dim = X.shape[-1]
    hidden_dim = 16
    num_layers = 2
    num_epochs = 300
    learning_rate = 1e-3
    num_classes = 2

    model = CommunityEvolutionLSTM(input_dim, hidden_dim, num_layers, num_classes).to(device)
    train_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, num_epochs, learning_rate, run, experiment_name)

    return train_losses, train_accs, val_accs, compute_accuracy(model, test_loader)




