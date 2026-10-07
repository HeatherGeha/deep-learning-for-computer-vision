import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def TrainModel(device: torch.device, num_epochs: int,
               model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, scheduler: torch.optim.lr_scheduler.LRScheduler,
               test_loader: torch.utils.data.DataLoader, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> torch.nn.Module:
        
    data = next(iter(train_loader))[0].to(device)
    print('Shape of the output from the convolutional part', model.convolutional(data).shape)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # TRAIN
        model.train()
        running_train_loss, correct_train, total_train = 0.0, 0, 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # VALIDATE
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1:02d} | LR: {current_lr:.6f} |"
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {100*epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {100*epoch_val_acc:.2f}%")
        scheduler.step()
    return model