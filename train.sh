#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=edu # The account name for the job.
#SBATCH --job-name=halfcheetah_medium_gpt # The job name.
#SBATCH -c 10 # The number of cpu cores to use.
#SBATCH --time=24:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=10gb # The memory the job will use per cpu core.

#SBATCH --gres=gpu
#SBATCH --constraint=k80

srun --pty -t 0-01:00 --gres=gpu:1 -A edu /bin/bash

module load anaconda

conda init bash

source ~/.bashrc

conda activate myenv

python train.py --config="configs/medium/halfcheetah_medium.yaml" --device="gpu" --seed="42"


# End of script
