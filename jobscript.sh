#!/bin/sh
# Submit with:  bsub -env "MODEL=latefusion,EPOCHS=20" < jobscript.sh
#BSUB -J videoclass
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 01:00
#BSUB -o videoclass_%J.out
#BSUB -e videoclass_%J.err

module load python3/3.11.9
source ~/venv_dlcv/bin/activate

python main.py --model ${MODEL:-latefusion} --epochs ${EPOCHS:-20} --data_dir /dtu/datasets1/02516/ufc10
