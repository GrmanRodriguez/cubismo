import os
import torch
from cubismo.model.unet import CFMUnet
from cubismo.policy.cfm_policy import CFMPolicy
from cubismo.utils.labels import Labels
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from io import BytesIO

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH","/checkpoint.ckpt")

app = FastAPI()
 
model = CFMUnet()
model.load_state_dict(torch.load(CHECKPOINT_PATH))
policy = CFMPolicy(model)

class GenerateRequest(BaseModel):
    label: str

@app.post("/generate")
def generate_image(request: GenerateRequest):
    label = request.label.upper()
    if label not in ["MAN", "WOMAN"]:
        raise HTTPException(status_code=400, detail="Label must be 'man' or 'woman'")

    image = policy.generate_images([Labels[label]])[0]

    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    return StreamingResponse(image_bytes, media_type="image/png")
