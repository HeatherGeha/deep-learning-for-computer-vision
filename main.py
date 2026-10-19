from datasets import GetFrameLoaders, GetVideoListLoaders, GetVideoStackLoaders, GetTwoStreamLoaders
from evaluator import TrainModel, TestModel
import aggregationnetwork
import earlyfusionnetwork
import latefusionnetwork
import conv3dnetwork
import dualstreamnetwork
import argparse
import torch
import os
import dualstreamnetworkV2
from datasets import GetSpatialV2Loaders, GetTemporalV2Loaders
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Each model file provides GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)
MODELS = {
    'aggregation': aggregationnetwork,
    'earlyfusion': earlyfusionnetwork,
    'latefusion': latefusionnetwork,
    'conv3d': conv3dnetwork,
    'twostream': dualstreamnetwork,
    'twostreamV2': dualstreamnetworkV2,
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
    parser.add_argument('--model', choices=MODELS.keys(), default='aggregation')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'ufc10'))
    # commented out path for Heather working locally
    # parser.add_argument('--data_dir',default=r'C:\Users\heath\Desktop\Heathers Desktop\school\DTU\deep learning in computer vision\project 2\ucf101 (1)\ufc10')
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

    elif args.model == "twostreamV2":
        import dualstreamnetworkV2
        from datasets import GetSpatialV2Loaders, GetTemporalV2Loaders

        job_id = os.environ.get("LSB_JOBID", "local")
        spatial_ckpt = f"spatial_stream_{job_id}.pth"
        temporal_ckpt = f"temporal_stream_{job_id}.pth"
        # STAGE 1 Spatial Stream
        print("\n" + "=" * 60)
        print(" [STAGE 1/3] TRAINING SPATIAL STREAM (APPEARANCE)")
        print("=" * 60)
        s_test_loader, s_train_loader, s_val_loader = GetSpatialV2Loaders(root_dir=root_dir, batch_size=32)

        spatial_net = dualstreamnetworkV2.SpatialStreamV2(num_classes=10, dropout=0.6).to(device)
        s_opt, s_crit, s_sched = dualstreamnetworkV2.GetSpatialOptimizerCriterionAndScheduler(spatial_net, num_epochs)

        spatial_net = TrainModel(device, num_epochs, spatial_net, s_opt, s_crit, s_sched, s_train_loader, s_val_loader)
        torch.save(spatial_net.state_dict(), spatial_ckpt)
        print(f"--> Saved spatial weights to {spatial_ckpt}")

        # STAGE 2 Temporal Stream 
   
        print("\n" + "=" * 60)
        print(" [STAGE 2/3] TRAINING TEMPORAL STREAM (DYNAMICS)")
        print("=" * 60)
        t_test_loader, t_train_loader, t_val_loader = GetTemporalV2Loaders(root_dir=root_dir, batch_size=8)

        temporal_net = dualstreamnetworkV2.TemporalStreamV2(num_classes=10, dropout=0.2).to(device)
        t_opt, t_crit, t_sched = dualstreamnetworkV2.GetTemporalOptimizerCriterionAndScheduler(temporal_net, num_epochs)

        temporal_net = TrainModel(device, num_epochs, temporal_net, t_opt, t_crit, t_sched, t_train_loader, t_val_loader)
        torch.save(temporal_net.state_dict(), temporal_ckpt)
        print(f"--> Saved temporal weights to {temporal_ckpt}")

        # STAGE 3
        print("\n" + "=" * 60)
        print(" [STAGE 3/3] COMBINING STREAMS & SWEEPING FUSION WEIGHTS")
        print("=" * 60)
        fused_model = dualstreamnetworkV2.DualStreamNetworkV2().to(device)
        fused_model.spatial_stream.load_state_dict(torch.load(spatial_ckpt, map_location=device))
        fused_model.temporal_stream.load_state_dict(torch.load(temporal_ckpt, map_location=device))

        best_alpha, test_acc = fused_model.evaluate_fusion_sweep(
            s_test_loader, 
            t_test_loader, 
            device=device, 
            step=0.05
        )
        print(f"Final Optimal Test Accuracy: {test_acc:.2f}% (at alpha={best_alpha:.2f})")
        torch.save(fused_model.state_dict(), f"twostreamV2_{job_id}_best.pth")

    else:
        model, optimizer, criterion, scheduler = MODELS[args.model].GetNetworkOptimizerCriterionAndScheduler(device, num_epochs)
        test_loader, train_loader, val_loader = GetVideoStackLoaders(root_dir)
        trained_model = TrainModel(device, num_epochs, model, optimizer, criterion, scheduler, train_loader, val_loader)
        _, test_acc = TestModel(device, trained_model, criterion, test_loader)
        print(f"Test Acc: {100*test_acc:.2f}%")
        torch.save(trained_model.state_dict(), f'{args.model}_best.pth')

    print("Model saved successfully!")
