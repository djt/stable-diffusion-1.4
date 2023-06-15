from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import os
import torch

# Download the model
def download_model():
    HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')

    repo = 'CompVis/stable-diffusion-v1-4'

    scheduler = DPMSolverMultistepScheduler.from_pretrained(repo, subfolder='scheduler') 
    model = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16, revision='fp16', scheduler=scheduler, use_auth_token=HF_AUTH_TOKEN, safety_checker=None)


if __name__ == "__main__":
    download_model()