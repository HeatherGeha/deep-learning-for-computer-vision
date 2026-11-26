import argparse
import torch
from pathlib import Path

from datasets import PotholesDataset
from proposals import GenerateAllProposals, SaveProposals, LoadProposals
from evaluate_proposals import EvaluateProposals
from dataloader import GetPotholeLoaders, GetPotholeLoadersMinusTest
from cnn import GetNetworkOptimizerCriterionAndScheduler
from evaluator import TrainCNNModel


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

# to use best proposals:                        python main.py --quality
# to use worse proposals:                       python main.py
# if you wish to generate other proposals:      python main.py --mode (choose mode) --size (optionally)

# note that selective search is deterministic, running the same mode/size combo twice will give identical proposals
   
def main():
    # run configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['fast', 'quality'], default='quality') # choose selective search mode
    parser.add_argument('--size', nargs=2, type=int, default=None) #optional resize images (uniform size=decreased runtime)
    parser.add_argument('--regenerate', action='store_true')
    args = parser.parse_args()

    #info for generating output file name
    size_name = 'original' if args.size is None else f'{args.size[0]}x{args.size[1]}'
    output_path = f'ss_{args.mode}_{size_name}.pt'

    size = tuple(args.size) if args.size else None

    # reuse existing proposals if available
    if Path(output_path).exists() and not args.regenerate:
        print(f'Loading existing proposals: {output_path}')
        results = LoadProposals(output_path)
    else:
        print(f'Generating proposals: {output_path}')
        results = GenerateAllProposals(mode=args.mode, size=size)

        # save as
        SaveProposals(results, output_path)

    #evaluate
    counts = [500, 750, 1000, 1250, 1500, 2000, 5000] # counts are how many proposals per image we want to evaluate
    #this range is something we can tune, choosing what works based on results

    # tuning found ideal counts: mode=fast counts=2000, mode=quality counts=5000
    EvaluateProposals(results, counts)

    num_epochs = 20

    # This uses a lot of ram so i dont load the test set.
    train_dataset = PotholesDataset(split='train', size=size)
    val_dataset = PotholesDataset(split='val', size=size)
    #test_dataset = PotholesDataset(split='test', size=size)

    #test_loader, train_loader, val_loader = GetPotholeLoadersMinusTest(train_dataset, val_dataset, test_dataset, results, batch_size=1, num_workers=0, max_proposals=500, outpute_image_size=(128, 128)) 
    train_loader, val_loader = GetPotholeLoadersMinusTest(train_dataset, val_dataset, results, batch_size=1, num_workers=0, max_proposals=500, outpute_image_size=(128, 128)) 
    device = GetDevice()
    model, optimizer, criterion, scheduler = GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)
    TrainCNNModel(device, num_epochs, model, optimizer, criterion, scheduler, train_loader, val_loader)


if __name__ == '__main__':
    main()