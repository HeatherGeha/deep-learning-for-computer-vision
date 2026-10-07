from datasets import GetFrameLoaders, GetVideoListLoaders, GetVideoStackLoaders
from evaluator import TrainModel
from conv3dnetwork import GetNetworkOptimizerCriterionAndScheduler
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    device = GetDevice()
    num_epochs = 2

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ufc10')

    test_loader, train_loader, val_loader = GetVideoStackLoaders(root_dir)
    model, optimizer, criterion, scheduler = GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)

    trained_model = TrainModel(device, num_epochs, model, optimizer, criterion, scheduler, test_loader, train_loader, val_loader)
    torch.save(trained_model.state_dict(), 'best_hotdog_cnn_32_64_128_1conv_dropout_lo.pth')
    print("Model saved successfully!")
