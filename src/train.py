import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import mobilenet_v2
from dataset import get_baseline_transforms, get_sota_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            
    val_loss = running_loss / total
    val_acc = correct.double() / total
    return val_loss, val_acc.item()

def main(use_sota=True, epochs=5):
    print(f"--- Starting Training (SOTA Augmentation: {use_sota}) ---")
    
    transform = get_sota_transforms() if use_sota else get_baseline_transforms()
    val_transform = get_baseline_transforms()

    if not os.path.exists("data/train") or len(os.listdir("data/train")) == 0:
        print("Error: 'data/train' is empty. Please add image folders for Happy, Neutral, Sad.")
        return

    train_dataset = datasets.ImageFolder("data/train", transform=transform)
    val_dataset = datasets.ImageFolder("data/val", transform=val_transform) if os.path.exists("data/val") else train_dataset

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = mobilenet_v2(weights="DEFAULT")
    model.classifier[1] = nn.Linear(model.last_channel, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/mobilenet_emotion.pth")

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main(use_sota=True)