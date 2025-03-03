#!/bin/bash
#SBATCH --job-name=knee-tunner
#SBATCH --output=logs/slurm/eval-%j.out 
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1              
#SBATCH --requeue                 # Enable automatic requeue on preemption
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --mail-user=wkong@scu.edu
#SBATCH --mail-type=END

LOGFILE=./logs/gpustats.log

(
  	while true; do
		echo "----- $(date) -----" >> $LOGFILE
		nvidia-smi >> $LOGFILE
		sleep 30
	done
) &
MONITOR_PID=$!

srun python3 -m src.evaluate --project_name xception_hyperband

kill $MONITOR_PID # Skeptical I'm doing this right, so manually kill process...		
