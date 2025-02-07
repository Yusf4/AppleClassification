import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define paths for training and test data
train_dir = "C:/Users/user/Desktop/appleClassification/train"
test_dir = "C:/Users/user/Desktop/appleClassification/test"

# Set device: Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define transforms for training (with augmentation) and testing (basic resize and normalization)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load datasets using ImageFolder (make sure your subfolders are named "apple" and "tomato")
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define a simple CNN for binary classification (apple vs tomato)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 3 channels, Output: 16 channels, kernel 3x3, padding to maintain size
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Reduces dimensions by half
        # Second convolution layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # After two poolings, with input size 224, spatial dims become: 224/2/2 = 56
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # Two classes: apple and tomato

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 -> relu -> pooling
        x = self.pool(F.relu(self.conv2(x)))  # conv2 -> relu -> pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model and move it to the appropriate device
model = SimpleCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights

        running_loss += loss.item()
        if (i + 1) % 20 == 0:  # Print every 20 batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}], Loss: {running_loss / 20:.3f}")
            running_loss = 0.0

print("Finished Training!")

# Evaluation on the Test Set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test images: {accuracy:.2f}%")
