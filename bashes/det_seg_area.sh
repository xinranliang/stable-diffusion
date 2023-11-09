# overall and by conditional prompts
python quality_metrics/det_seg.py --date 2023-10-12 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-12
python quality_metrics/det_seg.py --date 2023-10-26 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-26
python quality_metrics/det_seg.py --date 2023-11-03 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-03
python quality_metrics/det_seg.py --date 2023-11-05 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-05

# by domain label
python quality_metrics/detseg_bygender.py --date 2023-10-15 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15 --domain-name female
python quality_metrics/detseg_bygender.py --date 2023-10-15 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-15 --domain-name male
python quality_metrics/detseg_bygender.py --date 2023-10-29 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29 --domain-name female
python quality_metrics/detseg_bygender.py --date 2023-10-29 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-29 --domain-name male
python quality_metrics/detseg_bygender.py --date 2023-11-03 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-03 --domain-name female
python quality_metrics/detseg_bygender.py --date 2023-11-03 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-03 --domain-name male
python quality_metrics/detseg_bygender.py --date 2023-11-06 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06 --domain-name female
python quality_metrics/detseg_bygender.py --date 2023-11-06 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-06 --domain-name male

# generic prompt
python quality_metrics/det_seg.py --date 2023-10-30 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-30
python quality_metrics/det_seg.py --date 2023-10-31 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-31
python quality_metrics/det_seg.py --date 2023-11-01 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-01
python quality_metrics/det_seg.py --date 2023-11-02 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-02

python quality_metrics/detseg_bygender.py --date 2023-10-30 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-30 --domain-name female
python quality_metrics/detseg_bygender.py --date 2023-10-31 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-31 --domain-name female
python quality_metrics/detseg_bygender.py --date 2023-11-01 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-01 --domain-name female
python quality_metrics/detseg_bygender.py --date 2023-11-02 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-02 --domain-name female

python quality_metrics/detseg_bygender.py --date 2023-10-30 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-30 --domain-name male
python quality_metrics/detseg_bygender.py --date 2023-10-31 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-10-31 --domain-name male
python quality_metrics/detseg_bygender.py --date 2023-11-01 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-01 --domain-name male
python quality_metrics/detseg_bygender.py --date 2023-11-02 --master-folder /n/fs/xl-diffbia/projects/stable-diffusion/logs/samples/2023-11-02 --domain-name male
