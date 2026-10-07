## Introduction to Deep Learning in Computer Vision Project 2

### Models
One file per model, each with its own `GetNetworkOptimizerCriterionAndScheduler`:

| `--model` | File |
|---|---|
| `aggregation` | `aggregationnetwork.py` (trained on single frames, softmax averaged over the 10 frames at test time) |
| `earlyfusion` | `earlyfusionnetwork.py` |
| `latefusion` | `latefusionnetwork.py` |
| `conv3d` | `conv3dnetwork.py` |

### Running on the DTU HPC
One-time setup (on a GPU node, e.g. `voltash`). Use the cu126 build of torch: the default CUDA 13 build has no kernels for V100s.
```bash
module load python3/3.11.9
python3 -m venv ~/venv_dlcv
source ~/venv_dlcv/bin/activate
pip install pandas pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Quick test in an interactive session:
```bash
python main.py --model latefusion --epochs 1 --data_dir /dtu/datasets1/02516/ufc10
```

Batch job (output goes to `videoclass_<jobid>.out`):
```bash
bsub -env "MODEL=earlyfusion,EPOCHS=20" < jobscript.sh
```
