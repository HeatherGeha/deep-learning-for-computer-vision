from datasets_DRIVE import GetDRIVELoaders
from datasets_PH2 import GetPH2Loaders 
from evaluator import TrainModel, TestModel
import encdec
import argparse
import torch
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Each model file provides GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)
MODELS = {
    'encdec': encdec,
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
    parser.add_argument('--model', choices=MODELS.keys(), default='encdec')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ufc10'))
    # commented out path for Heather working locally
    # parser.add_argument('--data_dir',default=r'C:\Users\heath\Desktop\Heathers Desktop\school\DTU\deep learning in computer vision\project 2\ucf101 (1)\ufc10')
    args = parser.parse_args()

    device = GetDevice()
    num_epochs = args.epochs

    root_dir = args.data_dir

    model, optimizer, criterion, scheduler = MODELS[args.model].GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)
    test_loader, train_loader, val_loader = GetPH2Loaders(num_workers=0) #GetPH2Loaders(num_workers=0) #GetDRIVELoaders(num_workers=0)
    trained_model = TrainModel(device, num_epochs, model, optimizer, criterion, scheduler, train_loader, val_loader)
    _, test_acc = TestModel(device, trained_model, criterion, test_loader)
    print(f"Test Acc: {100*test_acc:.2f}%")
    torch.save(trained_model.state_dict(), f'{args.model}_best.pth')

    print("Model saved successfully!")
