import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def TrainCNNModel(device: torch.device, num_epochs: int,
                  model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, scheduler: torch.optim.lr_scheduler.LRScheduler,
                  train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> torch.nn.Module:

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # TRAIN
        model.train()
        running_loss, n_boxes, correct = 0.0, 0, 0
        n_potholes, correct_potholes = 0, 0

        for proposalImages, proposalLabels in train_loader:
            images = proposalImages.to(device)
            labels = proposalLabels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            predictions = output.argmax(1)

            running_loss += loss.item() * len(labels)
            correct += (predictions == labels).sum().item()
            n_boxes += len(labels)

            isPothole = labels == 1
            n_potholes += isPothole.sum().item()
            correct_potholes += (predictions[isPothole] == 1).sum().item()

        epoch_train_loss = running_loss / n_boxes
        epoch_train_acc = correct / n_boxes
        pothole_acc = correct_potholes / n_potholes if n_potholes > 0 else float('nan')
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)


        val_running_loss, val_boxes, val_correct = 0.0, 0, 0
        val_potholes, val_correct_potholes = 0, 0
        model.eval()
        with torch.no_grad():
            for proposalImages, proposalLabels in val_loader:
                images = proposalImages.to(device)
                labels = proposalLabels.to(device)

                output = model(images)
                loss = criterion(output, labels)

                predictions = output.argmax(1)

                val_running_loss += loss.item() * len(labels)
                val_correct += (predictions == labels).sum().item()
                val_boxes += len(labels)

                isPothole = labels == 1
                val_potholes += isPothole.sum().item()
                val_correct_potholes += (predictions[isPothole] == 1).sum().item()

            epoch_val_loss = val_running_loss / val_boxes
            epoch_val_acc = val_correct / val_boxes
            val_pothole_acc = val_correct_potholes / val_potholes if val_potholes > 0 else float('nan')
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)

        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:02d} | LR: {current_lr:.6f} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {100*epoch_train_acc:.2f}%, Train Pothole Acc: {100*pothole_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {100*epoch_val_acc:.2f}%, Val Pothole Acc: {100*val_pothole_acc:.2f}%")
        
        scheduler.step()
    return model

