from potassium import Potassium, Request, Response
from transformers import pipeline
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
#, DPMSolverMultistepScheduler
import base64
from io import BytesIO
import os

app = Potassium('my_app')

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print('Using device', device)
    
    HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')

    repo = 'CompVis/stable-diffusion-v1-4'
 
    # scheduler = DPMSolverMultistepScheduler.from_pretrained(repo, subfolder='scheduler')
    model = StableDiffusionPipeline.from_pretrained(repo, use_auth_token=HF_AUTH_TOKEN, safety_checker=None)
    model.enable_attention_slicing()
    model = model.to(device)

    context = {
        'model': model,
        # 'scheduler': scheduler
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    prompt = request.json.get('prompt', None)
    height = request.json.get('height', 512)
    width = request.json.get('width', 512)
    steps = request.json.get('steps', 50)
    guidance_scale = request.json.get('guidance_scale', 9)
    seed = request.json.get('seed', None)

    if not prompt: return Response(json={'message': 'No prompt was provided'}, status=500)

    generator = None
    if seed: generator = torch.Generator('cuda').manual_seed(seed)

    model = context.get('model')
    with autocast(device):
        image = model(prompt, guidance_scale=guidance_scale, height=height, width=width, num_inference_steps=steps, generator=generator).images[0]

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return Response(
        json = {'image_base64': image_base64}, 
        status=200
    )
    
if __name__ == '__main__':
    app.serve()