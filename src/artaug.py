"""ArtAug-inspired hard-case mining for facial emotion training data.

This is a lightweight, reproducible approximation of an iterative
synthesis-understanding loop. It is not the proprietary/official ArtAug system.
Only the training split is accepted to avoid validation/test leakage.
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from torchvision.transforms import functional as TF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Happy", "Neutral", "Sad"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def build_model(checkpoint):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model


def make_variants(image):
    """Create label-preserving low-light/backlight and pose stress variants."""
    variants = []
    low_light = ImageEnhance.Brightness(image).enhance(0.55)
    low_light = ImageEnhance.Contrast(low_light).enhance(0.8)
    variants.append(("low_light", low_light))

    backlit = ImageEnhance.Brightness(image).enhance(1.25)
    backlit = ImageEnhance.Contrast(backlit).enhance(0.65)
    variants.append(("backlight_washed", backlit))

    variants.append((
        "pose_left",
        TF.affine(image, angle=-12, translate=(8, 0), scale=0.95, shear=0, fill=(128, 128, 128)),
    ))
    variants.append((
        "pose_right",
        TF.affine(image, angle=12, translate=(-8, 0), scale=0.95, shear=0, fill=(128, 128, 128)),
    ))
    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Mine difficult training images and synthesize backlight/pose variants."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "models" / "baseline_mobilenet_emotion.pth"),
    )
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data" / "train"))
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "data" / "artaug_train")
    )
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if input_dir.name.lower() != "train":
        raise ValueError(
            "Hard-case mining must use data/train only. Do not mine validation or test data."
        )
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {checkpoint}. "
            "Run the baseline training experiment first."
        )

    dataset = datasets.ImageFolder(input_dir, transform=EVAL_TRANSFORM)
    if set(dataset.classes) != set(CLASSES):
        raise ValueError(f"Expected class folders: {CLASSES}; found {dataset.classes}")

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model = build_model(checkpoint)
    hard_cases = []
    offset = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1).cpu()
            for local_index, label in enumerate(labels.tolist()):
                hard_cases.append((
                    float(probabilities[local_index, label]),
                    offset + local_index,
                    int(label),
                    int(logits[local_index].argmax().item()),
                ))
            offset += len(labels)

    hard_cases.sort(key=lambda item: item[0])
    selected = hard_cases[:max(1, min(args.top_k, len(hard_cases)))]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "hard_cases_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["source_image", "true_label", "predicted_label", "true_class_probability", "variant", "output_image"])
        for confidence, index, true_label, predicted_label in selected:
            source_path, _ = dataset.samples[index]
            with Image.open(source_path) as opened:
                image = opened.convert("RGB")
            class_name = dataset.classes[true_label]
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for variant_name, variant_image in make_variants(image):
                output_path = class_dir / (
                    f"{Path(source_path).stem}_{variant_name}.jpg"
                )
                variant_image.save(output_path, quality=95)
                writer.writerow([
                    source_path,
                    class_name,
                    dataset.classes[predicted_label],
                    f"{confidence:.6f}",
                    variant_name,
                    output_path,
                ])

    print(f"Selected {len(selected)} difficult training images.")
    print(f"Generated {len(selected) * 4} targeted variants in: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(
        "Review the generated variants, then copy only approved class folders "
        "into data/train before retraining. Keep data/val and data/test untouched."
    )


if __name__ == "__main__":
    main()
