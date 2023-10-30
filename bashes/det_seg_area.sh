#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_2080:1          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/stable-diffusion/slurm_output/2023-10-xx/eval_detseg_area.txt            # where stdout and stderr will write to
#SBATCH -t 12:00:00           # time requested in hour:minute:second

source ~/.bashrc
conda activate latent-diffusion
cd /n/fs/xl-diffbia/projects/stable-diffusion

# overall and by conditional prompts
python quality_metrics/det_seg.py --date 2023-10-12 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12
python quality_metrics/det_seg.py --date 2023-10-26 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-26

# by domain label
python quality_metrics/detseg_bygender.py --date 2023-10-15 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15 --domain-name female
python quality_metrics/detseg_bygender.py --date 2023-10-15 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15 --domain-name male
python quality_metrics/detseg_bygender.py --date 2023-10-29 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/female --domain-name female
python quality_metrics/detseg_bygender.py --date 2023-10-29 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29/male --domain-name male