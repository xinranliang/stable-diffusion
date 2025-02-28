#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/stable-diffusion/slurm_output/2023-10-12/eval_is.txt            # where stdout and stderr will write to
#SBATCH -t 12:00:00           # time requested in hour:minute:second

source ~/.bashrc
conda activate latent-diffusion
cd /n/fs/xl-diffbia/projects/stable-diffusion

python quality_metrics/inception_score.py --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12