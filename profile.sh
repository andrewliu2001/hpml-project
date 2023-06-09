#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=edu # The account name for the job.
#SBATCH --job-name=halfcheetah_medium_gpt # The job name.
#SBATCH -c 10 # The number of cpu cores to use.
#SBATCH --time=11:00:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=10gb # The memory the job will use per cpu core.

#SBATCH --gres=gpu
#SBATCH --mem=120G
#SBATCH --constraint=k80


module load anaconda
conda init bash

source ~/.bashrc

conda activate myenv

python profile_hyena.py --config="configs/medium/halfcheetah_medium_hyena.yaml" --device="cuda" --seed="42"


# End of script
