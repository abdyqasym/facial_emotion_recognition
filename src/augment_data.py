import os
from pathlib import Path
from PIL import Image
from torchvision import transforms

# SOTA Augmentation pipeline
sota_transform = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])

def augment_directory(target_count=60):
    project_root = Path(__file__).resolve().parent.parent
    train_dir = project_root / "data" / "train"
    classes = ["Happy", "Neutral", "Sad"]

    print(f"Scanning dataset in: {train_dir}\n")

    for cls in classes:
        cls_dir = train_dir / cls
        if not cls_dir.exists():
            print(f"Directory {cls_dir} does not exist. Skipping...")
            continue

        # Get all valid image files
        valid_exts = (".jpg", ".jpeg", ".png", ".webp")
        all_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(valid_exts)]
        original_files = [f for f in all_files if not f.startswith("aug_")]

        if not original_files:
            print(f"⚠️ No original images found in {cls_dir}")
            continue

        print(f"Processing class '{cls}': found {len(original_files)} original images.")

        current_count = len(all_files)
        idx = 0
        while current_count < target_count:
            src_name = original_files[idx % len(original_files)]
            src_path = cls_dir / src_name

            try:
                img = Image.open(src_path).convert("RGB")
                aug_img = sota_transform(img)
                
                out_name = f"aug_{current_count}_{idx}.jpg"
                aug_img.save(cls_dir / out_name)
                current_count += 1
            except Exception as e:
                print(f"Error augmenting {src_name}: {e}")

            idx += 1

        print(f"✅ Class '{cls}' now has {len(os.listdir(cls_dir))} total images.\n")

if __name__ == "__main__":
    augment_directory(target_count=60)