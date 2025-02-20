import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image
from torchvision.transforms import transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = 2  # Assuming binary classification: 0 for apple, 1 for tomato
model = SimpleCNN(num_classes=num_classes)

# Load the state dictionary
model.load_state_dict(torch.load('apple_tomato_classifier.pth', map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = 'appleTesting.jpeg'
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0) # Add batch dimension
with torch.no_grad():
    output=model(input_tensor)
    probabilities=F.softmax(output,dim=1)
    predicted_class=torch.argmax(probabilities).item()

class_labels=["Apple","Tomato"]
print(f"Predicted: {class_labels[predicted_class]} (Confidence: {probabilities[0][predicted_class].item():.4f})")