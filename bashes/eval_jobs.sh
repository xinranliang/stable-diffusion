#!/bin/sh

#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=4     # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=24G             # memory 
#SBATCH --output=/n/fs/xl-diffbia/projects/stable-diffusion/slurm_output/2023-10-26/eval_jobs_repr.txt            # where stdout and stderr will write to
#SBATCH -t 6:00:00           # time requested in hour:minute:second

source ~/.bashrc
conda activate latent-diffusion
cd /n/fs/xl-diffbia/projects/stable-diffusion

# job_name w/o extended prompt
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-12 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-26 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-26

# job_name w/ extended prompt
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-15 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-15 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15 --domain-name male
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-29 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-29 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29 --domain-name male

# generic w/o extended prompt
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-30 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-30
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-31 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-31

# generic w/ extended prompt
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-30 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-30 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-31 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-31 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-30 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-30 --domain-name male
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-31 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-31 --domain-name male
