import os
import torch
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


SPECTROGRAM_DIR = "spectrograms"
MODEL_DIR = "models"

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")


class SpectrogramDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_dataset():
    image_paths = []
    labels = []

    folders = sorted([
        folder for folder in os.listdir(SPECTROGRAM_DIR)
        if os.path.isdir(os.path.join(SPECTROGRAM_DIR, folder))
    ])

    for folder in folders:
        folder_path = os.path.join(SPECTROGRAM_DIR, folder)

        if "_" in folder:
            label = folder.split("_", 1)[1]
        else:
            label = folder

        for file_name in os.listdir(folder_path):
            if not file_name.endswith(".png"):
                continue

            image_paths.append(
                os.path.join(folder_path, file_name)
            )

            labels.append(label)

    return image_paths, labels


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")

    image_paths, labels = load_dataset()

    print(f"Total images: {len(image_paths)}")
    print(f"Total labels: {len(set(labels))}")

    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    joblib.dump(
        encoder,
        os.path.join(MODEL_DIR, "cnn_label_encoder.pkl")
    )

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths,
        labels_encoded,
        test_size=0.25,
        random_state=42,
        stratify=labels_encoded
    )

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_dataset = SpectrogramDataset(
        X_train,
        y_train,
        transform
    )

    test_dataset = SpectrogramDataset(
        X_test,
        y_test,
        transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = SimpleCNN(
        num_classes=len(encoder.classes_)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nTraining started...\n")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels_batch in train_loader:
            images = images.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels_batch)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels_batch.size(0)

            correct += (predicted == labels_batch).sum().item()

        train_acc = 100 * correct / total

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {running_loss:.4f} "
            f"Train Accuracy: {train_acc:.2f}%"
        )

    print("\nEvaluating...\n")

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels_batch.size(0)

            correct += (predicted == labels_batch).sum().item()

    test_acc = 100 * correct / total

    print(f"\nTest Accuracy: {test_acc:.2f}%")

    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, "pokemon_cnn.pth")
    )

    print("\nCNN model saved:")
    print("models/pokemon_cnn.pth")


if __name__ == "__main__":
    train()