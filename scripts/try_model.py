import numpy as np
import torch
from cubismo.model.unet import CFMUnet
from cubismo.policy.cfm_policy  import CFMPolicy
from cubismo.utils.labels import Labels

CHECKPOINT_PATH = ""

model = CFMUnet()
model.load_state_dict(torch.load(CHECKPOINT_PATH))

policy = CFMPolicy(model)

labels = [Labels.MAN, Labels.WOMAN]

images = policy.generate_images(labels)

for image in images:
    image.show()