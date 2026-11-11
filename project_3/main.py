#from datasets_DRIVE import GetDRIVELoaders
from datasets_PH2 import GetPH2Loaders, GetPH2ClickLoaders
from evaluator import TrainModel as TrainFull, TestModel as TestFull
from evaluator_weak_annotation import TrainModel as TrainWeak, TestModel as TestWeak
import encdec
import argparse
import torch
import os
import random
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Each model file provides GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)
MODELS = {
    'encdec': encdec,
}

def SetSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def GetDevice() -> torch.device:
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        print("The code will run on Apple Silicon GPU (MPS).")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("The code will run on NVIDIA GPU (CUDA).")
        device = torch.device("cuda")
    else:
        print("The code will run on CPU.")
        device = torch.device("cpu")

    return device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS.keys(), default='encdec')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--supervision', choices=['full', 'weak'], default='full')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    device = GetDevice()
    num_epochs = args.epochs
    SetSeed(args.seed)

    model, optimizer, criterion, scheduler = MODELS[args.model].GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)
    if args.supervision == 'full':
        test_loader, train_loader, val_loader = GetPH2Loaders(num_workers=0)
        trained_model = TrainFull(device, num_epochs, model, optimizer, criterion, scheduler, train_loader, val_loader)
        _, test_acc = TestFull(device, trained_model, criterion, test_loader)

    else:
        test_loader, train_loader, val_loader = GetPH2ClickLoaders(num_workers=0, click_seed=args.seed)
        weak_criterion = encdec.PointLevelLoss()
        trained_model = TrainWeak(device, num_epochs, model, optimizer, weak_criterion, criterion, scheduler, train_loader, val_loader)
        _, test_acc = TestWeak(device, trained_model, criterion, test_loader)

    print(f"Test Acc: {100*test_acc:.2f}%")
    torch.save(trained_model.state_dict(), f'{args.model}_{args.supervision}_seed{args.seed}.pth')

    print("Model saved successfully!")