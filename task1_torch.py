# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomResizedCrop, ColorJitter
import random

# Set fixed seed for reproducibility
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)

# Setting Paths
PATH = os.path.abspath(os.getcwd())
IMG_PATH = os.path.join(PATH, 'images')
TRAIN_PATH = os.path.join(IMG_PATH, 'train')
TEST_PATH = os.path.join(IMG_PATH, 'test')
classes = ['kangaroo', 'yak']  # 0 is kangaroo, 1 is yak
mapping = {animal: i for i, animal in enumerate(classes)}
inverse_mapping = {i: animal for i, animal in enumerate(classes)}

# Displaying the dataset
for animal in classes:
    folder = os.path.join(TRAIN_PATH, animal)
    plt.figure(figsize=(10, 10))
    plt.title(animal.capitalize())
    for i, filename in enumerate(os.listdir(folder)[:9]):
        img_path = os.path.join(folder, filename)
        image = Image.open(img_path)
        plt.subplot(330 + 1 + i)
        plt.imshow(image)
        plt.axis('off')

    plt.show()


class AnimalDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []

        for subdir in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.endswith('.jpg'):
                        self.images.append(os.path.join(subdir_path, filename))
                        self.labels.append(mapping[subdir])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# Base transforms
base_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Augmentation transforms
aug_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# VGG1 architecture
class VGG1(nn.Module):
    def __init__(self):
        super(VGG1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 128 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# VGG3 architecture
class VGG3(nn.Module):
    def __init__(self):
        super(VGG3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# VGG16 Transfer Learning Model
class VGG16Transfer(nn.Module):
    def __init__(self, tune_all_layers=False):
        super(VGG16Transfer, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)

        # Freeze or unfreeze layers based on parameter
        for param in self.vgg16.parameters():
            param.requires_grad = tune_all_layers

        # Replace classifier
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.vgg16(x)


# MLP Model
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(256 * 256 * 3, 828),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(512, 512), nn.ReLU()) for _ in range(8)],
            *[nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(3)],
            *[nn.Sequential(nn.Linear(128, 128), nn.ReLU()) for _ in range(3)],
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


# Training function
def train_and_evaluate_model(model, train_loader, test_loader, model_name, num_epochs=20, device='cuda'):
    writer = SummaryWriter(f'runs/{model_name}_{time.time()}')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Print model summary
    print(model)

    start_time = time.time()
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()

            if i % 10 == 9:
                writer.add_scalar('training loss',
                                  running_loss / 10,
                                  epoch * len(train_loader) + i)
                writer.add_scalar('training accuracy',
                                  100 * correct / total,
                                  epoch * len(train_loader) + i)
                running_loss = 0.0

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1))
                test_loss += loss.item()
                predicted = (outputs > 0.5).float()
                test_total += labels.size(0)
                test_correct += (predicted.squeeze() == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        writer.add_scalar('test accuracy', test_acc, epoch)
        writer.add_scalar('test loss', test_loss / len(test_loader), epoch)

        history['loss'].append(running_loss / len(train_loader))
        history['accuracy'].append(100 * correct / total)
        history['val_loss'].append(test_loss / len(test_loader))
        history['val_accuracy'].append(test_acc)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Test Accuracy: {test_acc:.2f}%')

    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds')

    # Save the model
    torch.save(model.state_dict(), f'{model_name}.pth')

    return model, history

# Function to plot training history
def plot_history(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='test')
    plt.title('Cross Entropy Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='train')
    plt.plot(history['val_accuracy'], label='test')
    plt.title('Classification Accuracy')
    plt.legend()

    plt.suptitle(model_name)
    plt.savefig(f'{model_name}_history.png')
    plt.show()

# Create data loaders
def get_data_loaders(batch_size=32, augment=False):
    transform = aug_transform if augment else base_transform

    train_dataset = AnimalDataset(TRAIN_PATH, transform=transform)
    test_dataset = AnimalDataset(TEST_PATH, transform=base_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Training different models
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train VGG1
    train_loader, test_loader = get_data_loaders()
    vgg1_model = VGG1()
    vgg1_model, history = train_and_evaluate_model(
        vgg1_model, train_loader, test_loader, "VGG1", device=device)
    plot_history(history, "VGG1")

    # Train VGG3 without augmentation
    vgg3_model = VGG3()
    vgg3_model, history = train_and_evaluate_model(
        vgg3_model, train_loader, test_loader, "VGG3_no_aug", device=device)
    plot_history(history, "VGG3_no_aug")

    # Train VGG3 with augmentation
    train_loader_aug, test_loader = get_data_loaders(augment=True)
    vgg3_aug_model = VGG3()
    vgg3_aug_model, history = train_and_evaluate_model(
        vgg3_aug_model, train_loader_aug, test_loader, "VGG3_aug", device=device)
    plot_history(history, "VGG3_aug")

    # Train VGG16 with all layers tuned
    vgg16_all_layers = VGG16Transfer(tune_all_layers=True)
    vgg16_all_layers, history = train_and_evaluate_model(
        vgg16_all_layers, train_loader, test_loader, "VGG16_all_layers", device=device)
    plot_history(history, "VGG16_all_layers")

    # Train VGG16 with only MLP layers tuned
    vgg16_mlp = VGG16Transfer(tune_all_layers=False)
    vgg16_mlp, history = train_and_evaluate_model(
        vgg16_mlp, train_loader, test_loader, "VGG16_mlp", device=device)
    plot_history(history, "VGG16_mlp")

    # Train MLP model
    mlp_model = MLPModel()
    mlp_model, history = train_and_evaluate_model(
        mlp_model, train_loader, test_loader, "MLP", device=device)
    plot_history(history, "MLP")
