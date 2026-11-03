# Assignment 2: Facial Emotion Recognition

**Author:** Abdykassymov Abylay (Group SE-2420)  
**Classes:** Happy, Neutral, Sad  
**Backbone:** MobileNetV2 pretrained on ImageNet  
**Demo:** Streamlit + Grad-CAM

## Project overview

This project classifies facial-expression images into three classes and provides a Grad-CAM visualization. The app also includes a controlled Gaussian-noise robustness check. The noise check measures prediction consistency; it is not a video animation, VASA-1 implementation, accuracy, or F1-score evaluation.

The training script supports two comparable experiments:
- **Baseline:** resize, tensor conversion and ImageNet normalization.
- **Augmented:** the same model and training setup, with random rotation, horizontal flip, color jitter and affine transformations applied to training images.

These are data-augmentation experiments, not a claim that a new state-of-the-art architecture or method has been implemented.

## Dataset layout

Place the dataset in the following structure:

```text
data/
  train/
    Happy/
    Neutral/
    Sad/
  val/
    Happy/
    Neutral/
    Sad/
  test/
    Happy/
    Neutral/
    Sad/
```

All three splits are mandatory and must contain the same class folders. The script stops with an error if a split is missing or a class folder is absent. It never silently reuses the training set as validation data.

**Important split protocol:** split original subjects/images into train, validation and test sets *before* generating augmented copies. Keep all transformed copies of an original image in the training split only. Where subject identities are available, split by subject rather than by individual image to reduce identity leakage. Do not use the test split for tuning.

The dataset is excluded from Git. Document the dataset source, license, subject-level split procedure and class counts in the report.

## Installation

Use Python 3.10+ and install the dependencies:

```bash
pip install -r requirements.txt
```

## Training and evaluation

Run both experiments with the same fixed train/validation/test folders:

```bash
python src/train.py --experiment all --epochs 5
```

Or run one experiment:

```bash
python src/train.py --experiment baseline --epochs 5
python src/train.py --experiment augmented --epochs 5
```

The script sets random seeds, selects the best checkpoint by validation Macro-F1, then evaluates that checkpoint once on the held-out test set. It prints a per-class classification report and writes:
- `results/baseline_metrics.csv`
- `results/augmented_metrics.csv`

The augmented experiment also saves the app checkpoint to `models/mobilenet_emotion.pth`. Dataset splits must be identical between experiments for a fair comparison. For stronger reproducibility, record the Python/PyTorch/torchvision versions, hardware, dataset version and exact split manifest in the report.

## Run the application

Train the augmented model first so that `models/mobilenet_emotion.pth` exists, then run:

```bash
streamlit run app/app.py
```

The app now stops with a clear error if the trained checkpoint is missing or incompatible; it will not silently make predictions using random weights.

## Metrics and interpretation

- **Accuracy:** fraction of predictions matching independent ground-truth labels.
- **Macro-F1:** unweighted mean of per-class F1 scores; useful when class counts differ.
- **Weighted-F1:** mean of per-class F1 weighted by class support.
- **Prediction consistency:** fraction of noisy versions of one image that retain the clean image's predicted class. This is a robustness indicator only; it is not a ground-truth performance metric.

The app's noise robustness check uses 30 deterministic Gaussian perturbations of one input image. It does not synthesize video frames or implement VASA-1.

## Privacy

Images uploaded to Streamlit are processed by the machine hosting the app. In a remote deployment, the image is transmitted to that server, not processed on the visitor's own computer. Do not upload identifiable images without appropriate consent. The app does not intentionally persist uploaded images or create biometric identity profiles; deployment operators should separately review server logs, temporary-file handling and hosting-provider policies.

## Repository structure

- `src/train.py`: reproducible baseline/augmentation experiments and held-out test evaluation.
- `src/dataset.py`: training and evaluation transforms.
- `src/augment_data.py`: optional offline augmentation utility. Do not run this before splitting the original dataset.
- `app/app.py`: Streamlit interface and Grad-CAM / noise robustness workflow.
- `data/`: local dataset (not committed).
- `models/`: trained weights (not committed).
- `results/`: generated metric CSV files.
