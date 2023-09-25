import os
import subprocess

# 80 train split occupations
train_list = ['Actor', 'Architect', 'Audiologist', 'Author', 'Baker', 'Barber', 'Blacksmith', 'Bricklayer', 'Bus Driver', 'Butcher', 'Chef', 'Chemist', 'Cleaner', 'Coach', 'Comedian', 'Computer Programmer', 'Construction Worker', 'Consultant', 'Counselor', 'Dancer', 'Dentist', 'Designer', 'Dietitian', 'DJ', 'Doctor', 'Driver', 'Economist', 'Electrician', 'Engineer', 'Entrepreneur', 'Farmer', 'Florist', 'Graphic Designer', 'Hairdresser', 'Historian', 'Journalist', 'Judge', 'Lawyer', 'Librarian', 'Magician', 'Makeup Artist', 'Mathematician', 'Marine Biologist', 'Mechanic', 'Model', 'Musician', 'Nanny', 'Nurse', 'Optician', 'Painter', 'Pastry Chef', 'Pediatrician', 'Photographer', 'Plumber', 'Police Officer', 'Politician', 'Professor', 'Psychologist', 'Real Estate Agent', 'Receptionist', 'Recruiter', 'Researcher', 'Sailor', 'Salesperson', 'Surveyor', 'Singer', 'Social Worker', 'Software Developer', 'Statistician', 'Surgeon', 'Teacher', 'Technician', 'Therapist', 'Tour Guide', 'Translator', 'Vet', 'Videographer', 'Waiter', 'Writer', 'Zoologist']
# 20 test split occupations
test_list = ['Accountant', 'Astronaut', 'Biologist', 'Carpenter', 'Civil Engineer', 'Clerk', 'Detective', 'Editor', 'Firefighter', 'Interpreter', 'Manager', 'Nutritionist', 'Paramedic', 'Pharmacist', 'Physicist', 'Pilot', 'Reporter', 'Security Guard', 'Scientist', 'Web Developer']

# guidance values
w_lst = [1.0, 5.0, 9.0, 13.0, 17.0]

def get_job_prompt(job_name):
    return "A single {} in the center.".format(job_name.lower())


def guidance_sample():
    sys_comm = [
        "python", "scripts/txt2img.py", 
        "--outdir", "./logs", "--ddim_steps", "250", "--ddim_eta", "1.0", "--ckpt", "models/ldm/stable-diffusion-v1/sd-v1-4.ckpt", "--skip_grid",
        "--n_samples", "5", "--n_iter", "1"
    ] 
    for job_name in train_list + test_list:
        input_prompt = get_job_prompt(job_name)
        curr_sys_comm = sys_comm + ["--prompt", input_prompt]
        for w in w_lst:
            temp_sys_comm = curr_sys_comm + ["--scale", str(w)]
            subprocess.run(temp_sys_comm)


if __name__ == "__main__":
    guidance_sample()