python scripts/txt2img.py \
--prompt "a photograph of an astronaut riding a horse" \
--outdir ./logs \
--ddim_steps 250 --ddim_eta 1.0 \
--n_samples 4 --scale 1.0 \
--ckpt models/ldm/stable-diffusion-v1/sd-v1-4.ckpt