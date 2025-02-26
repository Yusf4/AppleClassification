import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define CNN Model
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

# Load Model
num_classes = 2
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load('apple_tomato_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = 'appleTesting.jpeg'
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform Prediction
with torch.no_grad():
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities).item()

print(f"Predicted: {['Apple', 'Tomato'][predicted_class]} (Confidence: {probabilities[0][predicted_class].item():.4f})")

# Function to Compute Grad-CAM
def compute_gradcam(model, input_tensor, class_idx):
    input_tensor.requires_grad = True
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    model.conv2.register_forward_hook(forward_hook)
    model.conv2.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * features, dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam / torch.max(cam)

    return cam

# Generate Grad-CAM Visualization
gradcam = compute_gradcam(model, input_tensor, predicted_class)
gradcam = gradcam.detach().numpy()

# Resize Grad-CAM to match the image size
gradcam = cv2.resize(gradcam, (224, 224))

# Convert to heatmap and ensure 3 channels
heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# Convert original PIL image to NumPy array
image = np.array(image.resize((224, 224)))  # Ensure same size as Grad-CAM

# Blend Heatmap and Original Image
overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

# Display Result
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(overlay)
plt.axis("off")

plt.show()
