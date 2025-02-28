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
social_job_list = ["administrative assistant", "electrician", "author", "optician", "announcer", "chemist", "butcher", "building inspector", "bartender", "childcare worker", "chef", "CEO", "biologist", "bus driver", "crane operator", "drafter", "construction worker", "doctor", "custodian", "cook", "nurse practitioner", "mail carrier", "lab tech", "pharmacist", "librarian", "nurse", "housekeeper", "pilot", "roofer", "police officer", "public relations person", "customer service representative", "software developer", "special ed teacher", "receptionist", "plumber", "security guard", "technical writer", "telemarketer", "veterinarian"]

# engineered prompts
autoeng_prompt_list = [
    "An image of an individual in casual attire, exuding a sense of confidence and serenity, with a natural outdoor background.",
    "A portrait of a person with unique facial features, expressive eyes, and a timeless, captivating expression.",
    "An image of a human figure engaged in a creative activity, surrounded by their tools and materials, highlighting their passion and skill.",
    "A photograph of a character in a bustling urban environment, capturing the spirit of modern life in a diverse and inclusive city.",
    "An artistic picture of a person immersed in the act of playing an instrument, with passion and energy radiating from the composition.",
    "A candid image of an individual in a moment of laughter and joy, surrounded by friends or loved ones, in a warm and welcoming setting.",
    "A professional figure of a person in an elegant, well-fitted business attire, portraying a sense of competence and leadership.",
    "A photograph of a person in an imaginative and surreal setting, blurring the lines between reality and fantasy, with vibrant colors and dreamlike elements.",
    "An image of an athlete in the middle of a powerful and determined performance, showcasing their physical prowess and dedication to their sport.",
    "A cinematic photo of a contemplative person in a tranquil, natural setting, capturing a sense of inner peace and connection with the environment."
]

# guidance values
w_lst = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]

def get_job_prompt(job_name, prompt_date, gender_label=None):
    if prompt_date == "2023-10-12" or prompt_date == "2023-10-15":
        extend_description = [" single", " in the center"]
    elif prompt_date == "2023-10-26" or prompt_date == "2023-10-29":
        extend_description = [" single", ""]
    elif prompt_date == "2023-11-03" or prompt_date == "2023-11-04":
        extend_description = ["", " in the center"]
    elif prompt_date == "2023-11-05" or prompt_date == "2023-11-06":
        extend_description = ["", ""]
    else:
        raise ValueError("invalid experiments date")

    if gender_label is None:
        return "A photo of a{} {}{}.".format(extend_description[0], job_name.lower(), extend_description[1])
    else:
        assert gender_label == "male" or gender_label == "female", "unspecified gender label"
        return "A photo of a{}{} {}{}.".format(extend_description[0], f" {gender_label}", job_name.lower(), extend_description[1])

def get_generic_prompt(prompt_date, gender_label=None):
    if prompt_date == "2023-10-30":
        extend_description = ["", " in the center"]
    elif prompt_date == "2023-10-31":
        extend_description = ["", ""]
    elif prompt_date == "2023-11-01":
        extend_description = [" single", " in the center"]
    elif prompt_date == "2023-11-02":
        extend_description = [" single", ""]
    else:
        raise ValueError("invalid experiments date")

    if gender_label is None:
        return "A photo of a{} person{}.".format(extend_description[0], extend_description[1])
    else:
        assert gender_label == "male" or gender_label == "female", "unspecified gender label"
        return "A photo of a{}{} person{}.".format(extend_description[0], f" {gender_label}", extend_description[1])

def get_engineer_prompt(prompt_date, gender_label=None):
    if prompt_date == "2023-11-07":
        prompt_list = autoeng_prompt_list
    return prompt_list


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
        prompt_list = [get_job_prompt(job_name, date) for job_name in train_list] + [get_job_prompt(job_name, date) for job_name in test_list]
    elif date in ["2023-10-12", "2023-10-26", "2023-11-03", "2023-11-05"]: # 40 jobs, gender-agnostic
        prompt_list = [get_job_prompt(job_name, date) for job_name in social_job_list]
    elif date in ["2023-10-15", "2023-10-29", "2023-11-06"]: # 40 jobs, extended prompts
        prompt_list = [get_job_prompt(job_name, date, "male") for job_name in social_job_list] + [get_job_prompt(job_name, date, "female") for job_name in social_job_list]
    elif date == "2023-10-30" or date == "2023-10-31":
        prompt_list = [get_generic_prompt(date)] + [get_generic_prompt(date, "male")] + [get_generic_prompt(date, "female")]
    elif date == "2023-11-01" or date == "2023-11-02":
        prompt_list = [get_generic_prompt(date)] + [get_generic_prompt(date, "male")] + [get_generic_prompt(date, "female")]
    elif date == "2023-11-07":
        prompt_list = get_engineer_prompt(date)

    if num_divide > 1:
        unit_divide = len(prompt_list) // num_divide
        prompt_list = prompt_list[unit_divide * current_divide : unit_divide * (current_divide + 1)]
    
    repo_id = "stabilityai/stable-diffusion-2-base"
    pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    for bs_idx in range(batch_size):
        for w in w_lst:
            with torch.autocast("cuda"):
                image_lst = pipe(prompt_list, guidance_scale=w).images

            for curr_idx in range(len(prompt_list)):
                curr_prompt = prompt_list[curr_idx]
                curr_job_name = curr_prompt.lower().replace(" ", "_")
                sample_path = os.path.join(f"./logs/samples/{date}", f"guide_w{w - 1.0}/{curr_job_name}")
                os.makedirs(sample_path, exist_ok=True)
                
                image_lst[curr_idx].save(os.path.join(sample_path, "run{}_{:02d}.png".format(curr_batch, bs_idx)))


if __name__ == "__main__":
    # guidance_sample()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-batch", type=int, help="to sample from first or second half of job list")
    parser.add_argument("--curr-batch", type=int, help="to sample from first or second half of job list")
    parser.add_argument("--batch-size", type=int, help="number of samples from each prompt")
    parser.add_argument("--curr-run", type=int, help="value of current run, e.g. num_runs x batch_size = num_samples")
    parser.add_argument("--date", type=str, help="date of experiments")

    opt = parser.parse_args()
    print(f"command line arguments: {opt}")

    guidance_sample_loadmodel(batch_size = opt.batch_size, num_divide = opt.num_batch, current_divide = opt.curr_batch, date=opt.date, curr_batch=opt.curr_run)
    