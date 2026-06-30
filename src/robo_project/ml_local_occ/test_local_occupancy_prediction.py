from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

try:
    from .model import SimpleLocalOccNet
except ImportError:
    from model import SimpleLocalOccNet


def load_input(sample_dir, image_size=64):
    cameras = ["front", "left", "right", "rear"]
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    imgs = []
    for cam in cameras:
        img = Image.open(sample_dir / f"{cam}_rgb.png").convert("RGB")
        imgs.append(tf(img))

    x = torch.cat(imgs, dim=0).unsqueeze(0)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample", default="sample_000000")
    parser.add_argument("--out", default="ml_runs/prediction_preview.png")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    sample_dir = dataset / args.sample

    if not sample_dir.exists():
        all_samples = sorted(dataset.glob("sample_*"))
        sample_dir = all_samples[0]
        print(f"Requested sample not found. Using {sample_dir.name}")

    device = torch.device("cpu")

    model = SimpleLocalOccNet().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = load_input(sample_dir).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0].cpu().numpy()
        pred = (prob > 0.5).astype(np.float32)

    gt = np.load(sample_dir / "local_occupancy_gt.npy").astype(np.float32)

    # For display:
    # GT: free=1, occupied=0, unknown=-1
    gt_display = gt.copy()
    gt_display[gt_display == -1] = 0.5

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.imshow(gt_display, vmin=0, vmax=1)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Prediction Probability")
    plt.imshow(prob, vmin=0, vmax=1)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Prediction Binary")
    plt.imshow(pred, vmin=0, vmax=1)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved preview: {out_path}")
    print(f"Sample used: {sample_dir.name}")


if __name__ == "__main__":
    main()
