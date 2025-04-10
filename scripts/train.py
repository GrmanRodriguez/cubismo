import argparse
import os
import random
import time
import torch
import torchvision
import wandb

from torch.utils.data import DataLoader
from tqdm import tqdm
from cubismo.model.unet import CFMUnet
from cubismo.policy.cfm_policy import CFMPolicy
from cubismo.utils.labels import Labels

def convert_label_to_gender_label(target):
    return target[20]

def collate_by_gender(batch):
    images, targets = zip(*batch)
    transformed_targets = [convert_label_to_gender_label(target) for target in targets]
    return torch.stack(images), torch.stack(transformed_targets)

def main(args):
    torch.cuda.empty_cache()

    training_start_time = time.time()

    if args.wandb:
        wandb.init(project=f"cubismo-celeba_{training_start_time:.0f}", config=vars(args))

    print("Creating dataset")
    dataset = torchvision.datasets.CelebA(
        root="./data",
        split="all",
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    )

    print("Creating model")
    model = CFMUnet()
    if args.checkpoint != "":
        print(f"Loading weights from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint))

    print("Creating policy")
    policy = CFMPolicy(model)

    print(f"Creating data loader ({args.batch_size=}, {args.loader_num_workers=})")
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=collate_by_gender
    )

    print(f"Setting up training params ({args.epochs=}, {args.lr=})")
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    checkpoints_base_name = f"celeba_{training_start_time:.0f}"
    amount_batches = len(train_loader)

    print("Here we go")
    policy.train()
    epoch_progress_bar = tqdm(total=args.epochs, desc="Epoch", position=0, leave=True)
    for epoch in range(args.epochs):
        batch_progress_bar = tqdm(total=amount_batches, desc="Batch", position=1, leave=True)
        epoch_loss = 0.0
        step = 0

        for images, labels in train_loader:
            policy.zero_grad()
            loss = policy.compute_loss(images.cuda(), labels.cuda())           
            if torch.isnan(loss):
                print("loss reached NaN. Early stopping.")
                raise ValueError("loss reached NaN")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            loss_value = loss.item()
            epoch_loss += loss_value
            batch_progress_bar.update(1)
            step += 1

            if args.wandb:
                wandb.log({
                    "step": step,
                    "batch_loss": loss_value,
                    "epoch": epoch + 1
                })

        scheduler.step()
        batch_progress_bar.close()

        avg_loss = epoch_loss / amount_batches
        current_lr = scheduler.get_last_lr()[0]

        if args.wandb:
            wandb.log({"epoch_avg_loss": avg_loss, "lr": current_lr, "epoch": epoch + 1})

        epoch_progress_bar.set_description(f"Epoch (Loss = {avg_loss:.6f})")
        epoch_progress_bar.update(1)

        if (epoch + 1) % args.eval_and_checkpoint_every == 0 or (epoch + 1) == args.epochs:
            checkpoint_directory = f"./checkpoints/{checkpoints_base_name}"
            checkpoint_name = f"{checkpoint_directory}/epoch={epoch+1}_loss={avg_loss:.6f}.ckpt"

            print(f"Saving {checkpoint_name}")
            os.makedirs(checkpoint_directory, exist_ok=True)
            torch.save(policy.weights(), checkpoint_name)

            print(f"Evaluating model at checkpoint {checkpoint_name}")
            evaluations_directory = f"./evaluations/{checkpoints_base_name}"
            evaluation_name = f"{evaluations_directory}/epoch={epoch+1}_loss={avg_loss:.6f}.jpg"

            print(f"Saving {evaluation_name}")
            policy.eval()
            os.makedirs(evaluations_directory, exist_ok=True)
            label = random.choice([Labels.MAN, Labels.WOMAN])
            image = policy.generate_images([label])[0]
            image.save(evaluation_name)

            if args.wandb:
                wandb.log({"sample_image": wandb.Image(str(evaluation_name), caption=f"Epoch {epoch+1}")})

            policy.train()

    batch_progress_bar.close()
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Cubismo model")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint to load (optional)")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--loader_num_workers", type=int, default=2, help="Number of workers for data loader")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=16, help="Number of training epochs")
    parser.add_argument("--eval_and_checkpoint_every", type=int, default=1, help="Frequency of evaluation and checkpointing")
    parser.add_argument("--wandb", type=bool, default=True, help="Enable Weights & Biases logging")

    args = parser.parse_args()
    main(args)