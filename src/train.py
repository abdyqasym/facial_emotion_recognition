import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def train_model():
  # Initialize MobileNetV2 with pretrained ImageNet weights
  model = mobilenet_v2(weights="DEFAULT")

  # Replace final classifier layer for 3 classes (Happy, Neutral, Sad)
  model.classifier[1] = nn.Linear(model.last_channel, 3)
  model = model.to(device)

  # Check if training data exists
  if not os.path.exists("data/train"):
    print("Warning: 'data/train' directory not found.")
    print(
        "Creating a dummy weight file for demo testing in 'models/' directory..."
    )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mobilenet_emotion.pth")
    print("Saved pretrained backbone weights to 'models/mobilenet_emotion.pth'")
    return

  # Load dataset
  train_dataset = datasets.ImageFolder("data/train", transform=transform)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Training loop (3 epochs)
  model.train()
  for epoch in range(3):
    running_loss = 0.0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    print(f"Epoch [{epoch+1}/3], Loss: {running_loss/len(train_loader):.4f}")

  os.makedirs("models", exist_ok=True)
  torch.save(model.state_dict(), "models/mobilenet_emotion.pth")
  print("Model successfully trained and saved to 'models/mobilenet_emotion.pth'")


if __name__ == "__main__":
  train_model()