import os
import subprocess
import argparse

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# 80 train split occupations
train_list = ['Actor', 'Architect', 'Audiologist', 'Author', 'Baker', 'Barber', 'Blacksmith', 'Bricklayer', 'Bus Driver', 'Butcher', 'Chef', 'Chemist', 'Cleaner', 'Coach', 'Comedian', 'Computer Programmer', 'Construction Worker', 'Consultant', 'Counselor', 'Dancer', 'Dentist', 'Designer', 'Dietitian', 'DJ', 'Doctor', 'Driver', 'Economist', 'Electrician', 'Engineer', 'Entrepreneur', 'Farmer', 'Florist', 'Graphic Designer', 'Hairdresser', 'Historian', 'Journalist', 'Judge', 'Lawyer', 'Librarian', 'Magician', 'Makeup Artist', 'Mathematician', 'Marine Biologist', 'Mechanic', 'Model', 'Musician', 'Nanny', 'Nurse', 'Optician', 'Painter', 'Pastry Chef', 'Pediatrician', 'Photographer', 'Plumber', 'Police Officer', 'Politician', 'Professor', 'Psychologist', 'Real Estate Agent', 'Receptionist', 'Recruiter', 'Researcher', 'Sailor', 'Salesperson', 'Surveyor', 'Singer', 'Social Worker', 'Software Developer', 'Statistician', 'Surgeon', 'Teacher', 'Technician', 'Therapist', 'Tour Guide', 'Translator', 'Vet', 'Videographer', 'Waiter', 'Writer', 'Zoologist']
# 20 test split occupations
test_list = ['Accountant', 'Astronaut', 'Biologist', 'Carpenter', 'Civil Engineer', 'Clerk', 'Detective', 'Editor', 'Firefighter', 'Interpreter', 'Manager', 'Nutritionist', 'Paramedic', 'Pharmacist', 'Physicist', 'Pilot', 'Reporter', 'Security Guard', 'Scientist', 'Web Developer']

# 44 occupations
social_job_list = ["administrative assistant", "electrician", "author", "optician", "announcer", "chemist", "butcher", "building inspector", "bartender", "childcare worker", "chef", "CEO", "biologist", "bus driver", "crane operator", "drafter", "construction worker", "doctor", "custodian", "cook", "nurse practitioner", "mail carrier", "lab tech", "pharmacist", "librarian", "nurse", "housekeeper", "pilot", "roofer", "police officer", "PR person", "customer service representative", "software developer", "special ed teacher", "receptionist", "plumber", "security guard", "technical writer", "telemarketer", "veterinarian"]

# guidance values
w_lst = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]

def get_job_prompt(job_name, gender_label=None):
    if gender_label is None:
        return "A photo of a single {} in the center.".format(job_name.lower())
    else:
        assert gender_label == "male" or gender_label == "female", "unspecified gender label"
        return "A photo of a single {} {} in the center.".format(gender_label, job_name.lower())


def guidance_sample():
    sys_comm = [
        "python", "scripts/txt2img.py", 
        "--outdir", "./logs", "--ddim_steps", "250", "--ddim_eta", "1.0", "--ckpt", "models/ldm/stable-diffusion-v1/sd-v1-4.ckpt", "--skip_grid",
        "--n_samples", "5", "--n_iter", "20"
    ] 
    for job_name in train_list + test_list:
        input_prompt = get_job_prompt(job_name)
        curr_sys_comm = sys_comm + ["--prompt", input_prompt]
        for w in w_lst:
            temp_sys_comm = curr_sys_comm + ["--scale", str(w)]
            subprocess.run(temp_sys_comm)


def guidance_sample_loadmodel(batch_size, num_divide, current_divide, date, curr_batch):
    if date == "2023-09-13": # 100 jobs
        prompt_list = [get_job_prompt(job_name) for job_name in train_list] + [get_job_prompt(job_name) for job_name in test_list]
    elif date == "2023-10-12": # 40 jobs, gender-agnostic
        prompt_list = [get_job_prompt(job_name) for job_name in social_job_list]
    elif date == "2023-10-15": # 40 jobs, extended prompts
        prompt_list = [get_job_prompt(job_name, "male") for job_name in social_job_list] + [get_job_prompt(job_name, "female") for job_name in social_job_list]

    unit_divide = len(prompt_list) // num_divide
    prompt_list = prompt_list[unit_divide * current_divide : unit_divide * (current_divide + 1)]
    
    repo_id = "stabilityai/stable-diffusion-2-base"
    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    for bs_idx in range(batch_size):
        for w in w_lst:
            image_lst = pipe(prompt_list, guidance_scale=w).images

            for curr_idx in range(len(prompt_list)):
                curr_prompt = prompt_list[curr_idx]
                curr_job_name = curr_prompt.lower().replace(" ", "_")
                sample_path = os.path.join(f"./logs/samples/{date}", f"guide_w{w - 1.0}/{curr_job_name}/run{curr_batch}")
                os.makedirs(sample_path, exist_ok=True)
                
                image_lst[curr_idx].save(os.path.join(sample_path, "{:02d}.png".format(bs_idx)))


if __name__ == "__main__":
    # guidance_sample()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-batch", type=int, help="to sample from first or second half of job list")
    parser.add_argument("--curr-batch", type=int, help="to sample from first or second half of job list")
    parser.add_argument("--batch-size", type=int, help="number of samples from each prompt")
    parser.add_argument("--curr-run", type=int, help="value of current run, e.g. num_runs x batch_size = num_samples")
    parser.add_argument("--date", type=str, help="date of experiments")

    opt = parser.parse_args()

    guidance_sample_loadmodel(batch_size = opt.batch_size, num_divide = opt.num_batch, current_divide = opt.curr_batch, date=opt.date, curr_batch=opt.curr_run)
    