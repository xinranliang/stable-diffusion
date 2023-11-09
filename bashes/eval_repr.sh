# job_name w/o extended prompt
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-12 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-26 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-26
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-11-03 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-03
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-11-05 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-05

# job_name w/ extended prompt
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-15 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-15 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15 --domain-name male
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-29 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-29 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29 --domain-name male
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-11-06 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-11-06 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06 --domain-name male

# generic w/o extended prompt
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-30 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-30
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-31 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-31
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-11-01 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-01
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-11-02 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-02

# generic w/ extended prompt
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-30 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-30 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-31 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-31 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-30 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-30 --domain-name male
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-10-31 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-31 --domain-name male
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-11-01 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-01 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-11-02 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-02 --domain-name female
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-11-01 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-01 --domain-name male
CUDA_VISIBLE_DEVICES=0 python domain_classifier/gender.py --date 2023-11-02 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-02 --domain-name male
