import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import mobilenet_v2

try:
    from dataset import get_baseline_transforms, get_sota_transforms
except ImportError:  # Allows: python -m src.train from the project root
    from src.dataset import get_baseline_transforms, get_sota_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPECTED_CLASSES = ["Happy", "Neutral", "Sad"]
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(dataset, shuffle=False):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def load_datasets(use_sota):
    """Load fixed train/val/test splits. Never substitute train data for validation."""
    required_dirs = {name: DATA_DIR / name for name in ("train", "val", "test")}
    missing = [str(path) for path in required_dirs.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Required dataset split(s) are missing: " + ", ".join(missing)
            + ". Create data/train, data/val and data/test with class subfolders."
        )

    train_transform = get_sota_transforms() if use_sota else get_baseline_transforms()
    eval_transform = get_baseline_transforms()

    train_dataset = datasets.ImageFolder(required_dirs["train"], transform=train_transform)
    val_dataset = datasets.ImageFolder(required_dirs["val"], transform=eval_transform)
    test_dataset = datasets.ImageFolder(required_dirs["test"], transform=eval_transform)

    expected = set(EXPECTED_CLASSES)
    for split_name, dataset in (
        ("train", train_dataset), ("val", val_dataset), ("test", test_dataset)
    ):
        classes = set(dataset.classes)
        if classes != expected:
            raise ValueError(
                f"{split_name} classes are {sorted(classes)}; expected "
                f"{sorted(expected)}. Class folders must match across all splits."
            )
        if len(dataset) == 0:
            raise ValueError(f"Dataset split '{split_name}' is empty.")

    if train_dataset.class_to_idx != val_dataset.class_to_idx or (
        train_dataset.class_to_idx != test_dataset.class_to_idx
    ):
        raise ValueError("Class-to-index mappings differ between dataset splits.")

    return (
        make_loader(train_dataset, shuffle=True),
        make_loader(val_dataset),
        make_loader(test_dataset),
        train_dataset.class_to_idx,
    )


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss, targets, predictions = 0.0, [], []

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())

    return total_loss / len(targets), accuracy_score(targets, predictions)


def evaluate(model, dataloader, criterion, class_names, print_report=False):
    model.eval()
    total_loss, targets, predictions = 0.0, [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())

    metrics = {
        "loss": total_loss / len(targets),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            targets, predictions, average="weighted", zero_division=0
        ),
    }
    if print_report:
        print(classification_report(
            targets, predictions, labels=list(range(len(class_names))),
            target_names=class_names, zero_division=0
        ))
    return metrics


def run_experiment(experiment_name, use_sota, epochs):
    # Reset initialization and data-order RNG so the two experiments start comparably.
    set_seed()
    print(f"\\n--- {experiment_name}: augmentation={'ON' if use_sota else 'OFF'} ---")
    train_loader, val_loader, test_loader, class_to_idx = load_datasets(use_sota)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    # Both experiments use the same ImageNet-pretrained MobileNetV2 architecture,
    # data split, optimizer, batch size and number of epochs.
    model = mobilenet_v2(weights="DEFAULT")
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_macro_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_metrics = evaluate(model, val_loader, criterion, class_names)
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model, test_loader, criterion, class_names, print_report=True
    )
    print(
        f"TEST | Accuracy: {test_metrics['accuracy']:.4f} | "
        f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
        f"Weighted-F1: {test_metrics['weighted_f1']:.4f}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "experiment": experiment_name,
        "augmentation": "on" if use_sota else "off",
        "epochs": epochs,
        "best_validation_macro_f1": best_val_macro_f1,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    result_path = RESULTS_DIR / f"{experiment_name}_metrics.csv"
    with result_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    if use_sota:
        checkpoint_path = MODEL_DIR / "mobilenet_emotion.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved app checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = MODEL_DIR / "baseline_mobilenet_emotion.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved baseline checkpoint for hard-case mining: {checkpoint_path}")

    print(f"Saved test metrics: {result_path}")
    return result


def main(experiment="all", epochs=5):
    set_seed()
    if epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    results = []
    if experiment in ("baseline", "all"):
        results.append(run_experiment("baseline", use_sota=False, epochs=epochs))
    if experiment in ("augmented", "all"):
        results.append(run_experiment("augmented", use_sota=True, epochs=epochs))

    if len(results) == 2:
        print("\\n--- Test-set comparison (same split and training setup) ---")
        for result in results:
            print(
                f"{result['experiment']}: "
                f"Accuracy={result['test_accuracy']:.4f}, "
                f"Macro-F1={result['test_macro_f1']:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and compare MobileNetV2 baseline and augmented models."
    )
    parser.add_argument(
        "--experiment", choices=("baseline", "augmented", "all"), default="all"
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    main(experiment=args.experiment, epochs=args.epochs)
