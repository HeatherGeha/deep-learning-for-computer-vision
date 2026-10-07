from datasets import GetFrameLoaders, GetVideoListLoaders, GetVideoStackLoaders
from evaluator import TrainModel, TestModel
import aggregationnetwork
import earlyfusionnetwork
import latefusionnetwork
import conv3dnetwork
import argparse
import torch
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Each model file provides GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)
MODELS = {
    'aggregation': aggregationnetwork,
    'earlyfusion': earlyfusionnetwork,
    'latefusion': latefusionnetwork,
    'conv3d': conv3dnetwork,
}

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
    parser.add_argument('--model', choices=MODELS.keys(), required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ufc10'))
    args = parser.parse_args()

    device = GetDevice()
    num_epochs = args.epochs

    root_dir = args.data_dir

    model, optimizer, criterion, scheduler = MODELS[args.model].GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)

    if args.model == 'aggregation':
        # Train and validate on single frames, test on whole videos by averaging the frame predictions
        _, train_loader, val_loader = GetFrameLoaders(root_dir)
        video_test_loader, _, _ = GetVideoStackLoaders(root_dir)
        trained_model = TrainModel(device, num_epochs, model, optimizer, criterion, scheduler, train_loader, val_loader)
        test_acc = aggregationnetwork.EvaluateAggregated(device, trained_model, video_test_loader)
    else:
        test_loader, train_loader, val_loader = GetVideoStackLoaders(root_dir)
        trained_model = TrainModel(device, num_epochs, model, optimizer, criterion, scheduler, train_loader, val_loader)
        _, test_acc = TestModel(device, trained_model, criterion, test_loader)

    print(f"Test Acc: {100*test_acc:.2f}%")
    torch.save(trained_model.state_dict(), f'{args.model}_best.pth')
    print("Model saved successfully!")
