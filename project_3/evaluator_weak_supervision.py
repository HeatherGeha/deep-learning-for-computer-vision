import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Helper function to check if its a tuple or a list
def _to_device(data, device):
    if isinstance(data, (tuple, list)):
        return [x.to(device) for x in data]
    return data.to(device)


def TrainModel(device: torch.device, num_epochs: int,
               model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, val_criterion: torch.nn.Module, scheduler: torch.optim.lr_scheduler.LRScheduler,
               train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> torch.nn.Module:

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss, best_state = float('inf'), None

    for epoch in range(num_epochs):
        # TRAIN
        model.train()
        running_loss, n_images, correct, n_pixels = 0.0, 0, 0, 0

        for images, labels in train_loader:
            images = _to_device(images, device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            n_images += labels.size(0)
            preds = (outputs > 0).float()
            labelled = labels != -1
            correct += ((preds == labels) & labelled).sum().item()
            n_pixels += labelled.sum().item()

        epoch_train_loss = running_loss / n_images
        epoch_train_acc = correct / n_pixels
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # VALIDATE
        epoch_val_loss, epoch_val_acc = TestModel(device, model, val_criterion, val_loader)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:02d} | LR: {current_lr:.6f} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {100*epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {100*epoch_val_acc:.2f}%")

        # Keep the weights from the epoch with the lowest validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    print(f"Best Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_state)
    return model


def TestModel(device, model, criterion, loader):
    """Returns (loss, pixel accuracy) of the model on the given loader."""
    model.eval()
    running_loss, n_images, correct, n_pixels = 0.0, 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = _to_device(images, device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            n_images += labels.size(0)
            preds = (outputs > 0).float()
            correct += (preds == labels).sum().item()
            n_pixels += labels.numel()

    return running_loss / n_images, correct / n_pixels


"""Full 5 metric evaluator"""

def EvaluateAllMetrics(device, model, criterion, loader, eps=1e-7):
    """
    Dice Overlap, IoU, Accuracy, Sensitivity, and Specificity.
    """
    model.eval()
    running_loss, n_images = 0.0, 0
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = _to_device(images, device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            n_images += labels.size(0)

            preds = (outputs > 0).float()
            total_tp += (preds * labels).sum().item()
            total_fp += (preds * (1 - labels)).sum().item()
            total_fn += ((1 - preds) * labels).sum().item()
            total_tn += ((1 - preds) * (1 - labels)).sum().item()

    avg_loss = running_loss / n_images
    tp, fp, fn, tn = total_tp, total_fp, total_fn, total_tn

    """we now have the four metrics true positive, false positive, false negative and true negative, and we can use them to build our 5 metrics"""

    return {
        'loss': avg_loss,
        'dice': (2 * tp) / (2 * tp + fp + fn + eps),
        'iou': tp / (tp + fp + fn + eps),
        'accuracy': (tp + tn) / (tp + tn + fp + fn + eps),
        'sensitivity': tp / (tp + fn + eps),
        'specificity': tn / (tn + fp + eps),
    } 