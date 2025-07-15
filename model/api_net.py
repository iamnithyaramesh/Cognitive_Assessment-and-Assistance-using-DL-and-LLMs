import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Image preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os

# Get absolute path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

resnet_checkpoint_path = os.path.join(BASE_DIR, 'Resnet_Backbone_Checkpoint.pth')
api_checkpoint_path = os.path.join(BASE_DIR, 'API_Net_Checkpoint.pth')

resnet_checkpoint = torch.load(resnet_checkpoint_path, map_location=device)

# Define APIComponent
class APIComponent(nn.Module):
    def __init__(self, feature_dim):
        super(APIComponent, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x1, x2):
        xm = torch.cat((x1, x2), dim=1)
        xm = self.mlp(xm)
        g1 = torch.sigmoid(xm * x1)
        g2 = torch.sigmoid(xm * x2)
        x1_self = x1 + x1 * g1
        x2_self = x2 + x2 * g2
        x1_other = x1 + x1 * g2
        x2_other = x2 + x2 * g1
        return x1_self, x1_other, x2_self, x2_other

# Define Classifier
class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Define APINet
class APINet(nn.Module):
    def __init__(self, backbone, api_component, classifier):
        super(APINet, self).__init__()
        self.backbone = backbone
        self.api_component = api_component
        self.classifier = classifier

    def forward(self, x1, x2):
        x1 = self.backbone(x1).squeeze(-1).squeeze(-1)
        x2 = self.backbone(x2).squeeze(-1).squeeze(-1)
        x1_self, x1_other, x2_self, x2_other = self.api_component(x1, x2)
        p1_self = self.classifier(x1_self)
        p1_other = self.classifier(x1_other)
        p2_self = self.classifier(x2_self)
        p2_other = self.classifier(x2_other)
        return p1_self, p1_other, p2_self, p2_other

# Load ResNet-50 and initialize as backbone
resnet = models.resnet50(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 6)
resnet.load_state_dict(resnet_checkpoint['model_state_dict'])
resnet.to(device)

# Remove last layer to use as backbone
backbone = nn.Sequential(*list(resnet.children())[:-1])
backbone.to(device)

# Initialize API-Net
api_component = APIComponent(feature_dim=2048)
classifier = Classifier(feature_dim=2048, num_classes=6)
model = APINet(backbone, api_component, classifier)
model.to(device)

api_checkpoint = torch.load(api_checkpoint_path, map_location=device)
model.load_state_dict(api_checkpoint['model_state_dict'])

# Prediction function for local image
def predict_local_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = data_transforms(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            p1_self, _, _, _ = model(image, image)
            _, predicted = torch.max(p1_self, 1)
        return predicted.item()

    except Exception as e:
        print(f"Error processing local image: {e}")
        return None

'''# Predict from test.jpg (must be in the same directory)
local_image_path = "test0.png"
predicted_class = predict_local_image(local_image_path)

if predicted_class is not None:
    print(f'The predicted class for the clock image is: {predicted_class}')
else:
    print("Could not predict the class for the local image.")

# Predict from test.jpg (must be in the same directory)
local_image_path = "test1.png"
predicted_class = predict_local_image(local_image_path)

if predicted_class is not None:
    print(f'The predicted class for the clock image is: {predicted_class}')
else:
    print("Could not predict the class for the local image.")

    # Predict from test.jpg (must be in the same directory)
local_image_path = "test2.png"
predicted_class = predict_local_image(local_image_path)

if predicted_class is not None:
    print(f'The predicted class for the clock image is: {predicted_class}')
else:
    print("Could not predict the class for the local image.")

    # Predict from test.jpg (must be in the same directory)
local_image_path = "test3.png"
predicted_class = predict_local_image(local_image_path)

if predicted_class is not None:
    print(f'The predicted class for the clock image is: {predicted_class}')
else:
    print("Could not predict the class for the local image.")

    # Predict from test.jpg (must be in the same directory)
local_image_path = "test4.png"
predicted_class = predict_local_image(local_image_path)

if predicted_class is not None:
    print(f'The predicted class for the clock image is: {predicted_class}')
else:
    print("Could not predict the class for the local image.")

    # Predict from test.jpg (must be in the same directory)
local_image_path = "test5.png"
predicted_class = predict_local_image(local_image_path)

if predicted_class is not None:
    print(f'The predicted class for the clock image is: {predicted_class}')
else:
    print("Could not predict the class for the local image.")'''


import cv2
import numpy as np

def clean_and_normalize_clock(input_path, output_path=None):
    # Load the image in grayscale
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Invert image to make pencil lines darker
    inverted = 255 - gray

    # Use dilation to bold the lines
    kernel = np.ones((2, 2), np.uint8)
    bold = cv2.dilate(inverted, kernel, iterations=1)

    # Invert back to original color scheme
    final = 255 - bold

    if output_path:
        cv2.imwrite(output_path, final)

    return final



local_image_path = "img.png"  # Path to the processed image
predicted_class = predict_local_image(local_image_path)

if predicted_class is not None:
    print(f'The predicted class for the clock image is: {predicted_class}')
else:
    print("Could not predict the class for the local image.")