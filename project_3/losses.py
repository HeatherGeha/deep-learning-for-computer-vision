import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        # Map raw logits (-inf, +inf) to valid probabilities [0, 1]
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice_coeff = (2.0 * intersection + self.eps) / (probs.sum() + targets.sum() + self.eps)
        
        return 1.0 - dice_coeff


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_t * focal_weight
            
        return (focal_weight * bce_loss).mean()


class DiceFocalLoss(nn.Module):
    """
    Hybrid loss combining Dice Loss and Focal Loss.
    """
    def __init__(self, dice_weight=1.0, focal_weight=1.0, alpha=0.25, gamma=2.0, eps=1e-7):
        super().__init__()
        self.dice = DiceLoss(eps=eps)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        loss_dice = self.dice(logits, targets)
        loss_focal = self.focal(logits, targets)
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal

# loss function for point-level supervision
# L: two classes, Lesion and Background
# L': is empty, because we have both classes present in every image
# I_s are the pixels containing clicks
# G_i is the click label (0, no click or 1, click)
# alpha_i is the click weight
# Point-level-loss = Image-level-loss(S,L,L')- sum over all i in I(alpha_i log(S_{iG_i}))
class PointLevelLoss(nn.Module): 
    def __init__(self, alpha=1.0): # default alpha=1 is the same as regular cross-entropy on clicked pixels only
        super().__init__()
        self.alpha = alpha
     
    def forward(self, predictions, targets):
        # implement the image level loss
        # convert logits into lesion probabilities (networks output logits)
        probabilities = torch.sigmoid(predictions)
        # find the highest probability for each class
        max_lesion = probabilities.flatten(1).max(dim=1).values
        max_background = (1 - probabilities).flatten(1).max(dim=1).values
        # calculate image-level loss
        image_loss = -0.5 * (torch.log(max_lesion.clamp(min=1e-7)).mean()+ torch.log(max_background.clamp(min=1e-7)).mean())

        # now we implement the click-term (the subtracted term in point-level-loss)
        # identify labelled pixels
        labelled = targets != -1
        # replace unlabelled targets with 0
        targets = targets.clamp(min=0)
        # calculate click-level loss
        click_loss = -(targets * torch.log(probabilities.clamp(min=1e-7))
        + (1 - targets) * torch.log((1 - probabilities).clamp(min=1e-7)))
        # keep only the losses at clicked pixels
        click_loss = click_loss * labelled
        # average over clicked pixels
        click_loss = click_loss.flatten(1).sum(dim=1).mean()

        return image_loss + self.alpha * click_loss
