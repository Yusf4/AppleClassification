import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image  # FIXED import
import torchvision.transforms as transforms  # FIXED import

# Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # Ensure this matches training size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Model
num_classes = 2  # 0: Apple, 1: Tomato
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load('apple_tomato_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# Define Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure it matches training input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and Preprocess Image
image_path = 'appleTesting.jpeg'
image = Image.open(image_path).convert("RGB")  # Convert to RGB to avoid grayscale issues
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

# Perform Prediction
with torch.no_grad():
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
    predicted_class = torch.argmax(probabilities).item()

# Define Class Labels
class_labels = ["Apple", "Tomato"]
print(f"Predicted: {class_labels[predicted_class]} (Confidence: {probabilities[0][predicted_class].item():.4f})")
