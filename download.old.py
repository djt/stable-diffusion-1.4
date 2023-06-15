from diffusers import StableDiffusionPipeline
import os

def download_model():
    HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')

    repo = 'CompVis/stable-diffusion-v1-4'
 
    model = StableDiffusionPipeline.from_pretrained(repo, use_auth_token=HF_AUTH_TOKEN, safety_checker=None)

if __name__ == "__main__":
    download_model()