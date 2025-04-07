import argparse
import math
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch

from cubismo.model.unet import CFMUnet
from cubismo.policy.cfm_policy import CFMPolicy
from cubismo.utils.labels import Labels

def load_policy(checkpoint_path):
    model = CFMUnet()
    model.load_state_dict(torch.load(checkpoint_path))
    policy = CFMPolicy(model)
    return policy

def generate_and_display_images(checkpoint_path, num_images):
    policy = load_policy(checkpoint_path)
    policy.eval()

    labels = [random.choice([Labels.MAN, Labels.WOMAN]) for _ in range(num_images)]

    images = policy.generate_images(labels)

    grid_size = math.ceil(math.sqrt(num_images))

    _, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
    axes = axes.flatten()  # Make it easier to iterate over axes

    for idx, ax in enumerate(axes):
        if idx < num_images:
            ax.imshow(images[idx])
            label_text = "Man" if labels[idx] == Labels.MAN else "Woman"
            ax.set_title(label_text)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Try a cubismo model.")
    parser.add_argument("--checkpoint", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--num_images", type=int, nargs="?", default=9, help="Number of images to generate.")
    args = parser.parse_args()

    generate_and_display_images(args.checkpoint, args.num_images)