from datasets_DRIVE import GetDRIVELoaders, GetAugmentedDRIVELoaders
from datasets_PH2 import GetPH2Loaders 
from evaluator import EvaluateAllMetrics, TrainModel, TestModel
from losses import DiceLoss
from encdec import Network as EncDecNetwork
from unet import UNet
import argparse
import encdec
import unet
import extendedunet
import torch
import os
import random
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Each model file provides GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)
MODELS = {
    'encdec': encdec,
    'unet': unet,
    'unet_hybridloss': extendedunet,
    'extendedunet': extendedunet,
}

def set_seed(seed: int):
    """Sets deterministic seeds across PyTorch, NumPy, and Python built-ins."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def compute_mean_std(results_list):
    """Computes mean and std across multiple seed evaluation runs."""
    keys = results_list[0].keys()
    summary = {}
    for k in keys:
        vals = [r[k] for r in results_list]
        summary[k] = (np.mean(vals), np.std(vals))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS.keys(), default='encdec')
    parser.add_argument('--dataset', choices=['PH2', 'DRIVE'], default='PH2')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ufc10'))
    # commented out path for Heather working locally
    # parser.add_argument('--data_dir',default=r'C:\Users\heath\Desktop\Heathers Desktop\school\DTU\deep learning in computer vision\project 2\ucf101 (1)\ufc10')
    args = parser.parse_args()

    device = GetDevice()
    num_epochs = args.epochs

    all_train, all_val, all_test = [], [], []

    for seed in args.seeds:
        print(f"\n>>> Running Seed {seed} for Model: {args.model.upper()} on Dataset: {args.dataset} <<<")
        set_seed(seed)

        model, optimizer, criterion, scheduler = MODELS[args.model].GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)

        if args.dataset == 'PH2':
            test_loader, train_loader, val_loader = GetPH2Loaders(num_workers=0)
        elif args.model == 'extendedunet':
            test_loader, train_loader, val_loader = GetAugmentedDRIVELoaders(num_workers=0)
        else:
            test_loader, train_loader, val_loader = GetDRIVELoaders(num_workers=0)

        # Train
        trained_model = TrainModel(device, num_epochs, model, optimizer, criterion, scheduler, train_loader, val_loader)
        _, test_acc = TestModel(device, trained_model, criterion, test_loader)
        print(f"Seed {seed} Test Acc: {100*test_acc:.2f}%")
        torch.save(trained_model.state_dict(), f'{args.model}_{args.dataset}_seed{seed}_best.pth')

        # Collect 5 metrics 
        all_train.append(EvaluateAllMetrics(device, trained_model, criterion, train_loader))
        all_val.append(EvaluateAllMetrics(device, trained_model, criterion, val_loader))
        all_test.append(EvaluateAllMetrics(device, trained_model, criterion, test_loader))

    # Compute mean 
    train_stats = compute_mean_std(all_train)
    val_stats = compute_mean_std(all_val)
    test_stats = compute_mean_std(all_test)

    # Stats
    print('\n' + '=' * 75)
    print(f'   AGGREGATE 5-METRIC REPORT ({args.model.upper()} on {args.dataset} over seeds {args.seeds})')
    print('=' * 75)
    print(f"{'Split':<8} | {'Loss':<12} | {'Dice':<12} | {'IoU':<12} | {'Acc':<12} | {'Sens':<12} | {'Spec':<12}")
    print('-' * 75)

    for split_name, stats in [('Train', train_stats), ('Val', val_stats), ('Test', test_stats)]:
        def fmt(m):
            return f"{m[0]:.3f}±{m[1]:.3f}"

        print(
            f"{split_name:<8} | {fmt(stats['loss']):<12} | {fmt(stats['dice']):<12} | {fmt(stats['iou']):<12} | "
            f"{fmt(stats['accuracy']):<12} | {fmt(stats['sensitivity']):<12} | {fmt(stats['specificity']):<12}"
        )
    print('=' * 75 + '\n')