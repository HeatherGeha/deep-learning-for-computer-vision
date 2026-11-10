#!/bin/sh
### ------------- specify queue name ----------------
#BSUB -q c02516
### ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
### ------------- specify job name ----------------
#BSUB -J task3_segmentation
### ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=10GB]"
### ------------- specify wall-clock time (45 mins for 3-seed runs) ----------------
#BSUB -W 00:45
### ------------- log files ----------------
#BSUB -o run_%J.out
#BSUB -e run_%J.err

# Load Python environment
module load python3/3.11.9
source ~/venv_02516/bin/activate

# Move into the project_3 directory
cd ~/deep-learning-for-computer-vision/project_3

# Default fallback values if variables are not passed via bsub -env
MODEL=${MODEL:-unet}
DATASET=${DATASET:-PH2}
EPOCHS=${EPOCHS:-25}
SEEDS=${SEEDS:-"42 43 44"}

echo "Running Model: ${MODEL} | Dataset: ${DATASET} | Epochs: ${EPOCHS} | Seeds: ${SEEDS}"

# Execute training and evaluation script
python main.py --model ${MODEL} --dataset ${DATASET} --epochs ${EPOCHS} --seeds ${SEEDS}