#!/bin/bash
#SBATCH --job-name=knee
#SBATCH --output=logs/slurm/resnet50_hyperband-%j.out 
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              
#SBATCH --requeue                 # Enable automatic requeue on preemption
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=wkong@scu.edu
#SBATCH --mail-type=END

# LOGFILE=./logs/gpustats.log

# (
#   	while true; do
# 		echo "----- $(date) -----" >> $LOGFILE
# 		nvidia-smi >> $LOGFILE
# 		sleep 300
# 	done
# ) &
# MONITOR_PID=$!

srun python3 -m src.train --config ./configs/resnet50_hyperband.yaml

# kill $MONITOR_PID # Skeptical I'm doing this right, so manually kill process...		
