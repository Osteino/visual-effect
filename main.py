import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VisualEffectModel(nn.Module):
    def __init__(self, base_model):
        super(VisualEffectModel, self).__init__()
        # Extract only the feature layers we need
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        
    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        outputs['layer1'] = x
        
        x = self.layer2(x)
        outputs['layer2'] = x
        
        x = self.layer3(x)
        outputs['layer3'] = x
        
        return outputs

def load_image(image_path, size=224):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def visual_effect(model, image, num_iterations=100, lr=0.01):
    model = model.to(device)
    input_tensor = image.to(device).requires_grad_(True)
    optimizer = optim.Adam([input_tensor], lr=lr)
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        out = model(input_tensor)
        
        # Calculate loss using feature activations
        loss = 0
        for layer_output in out.values():
            loss += torch.mean(layer_output)
        
        loss.backward()
        
        # Normalize gradients
        grad = input_tensor.grad.data / (input_tensor.grad.data.std() + 1e-8)
        input_tensor.data += lr * grad
        input_tensor.grad.data.zero_()
    
    return input_tensor.detach().cpu()

def main():
    # Step 1 - download a pre-trained PyTorch model
    url = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'  # Example PyTorch model
    data_dir = './data/'
    model_name = os.path.split(url)[-1]
    local_model_file = os.path.join(data_dir, model_name)

    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(local_model_file):
        # Download the model
        model_url = urllib.request.urlopen(url)
        with open(local_model_file, 'wb') as output:
            output.write(model_url.read())

    # Step 2 - Load the PyTorch model
    model = models.resnet18(weights='IMAGENET1K_V1')  # Use pre-trained weights
    
    # Create our custom model with the base features
    effect_model = VisualEffectModel(model)
    effect_model.eval()

    # Print model details
    layers = [name for name, _ in effect_model.named_modules() if isinstance(_, torch.nn.Conv2d)]
    feature_nums = [module.out_channels for module in effect_model.modules() if isinstance(module, torch.nn.Conv2d)]

    print('Number of layers:', len(layers))
    print('Total number of feature channels:', sum(feature_nums))

    # Example: Start with a random noise image
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

    # Helper functions for visualization
    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1) * 255)
        plt.imshow(a)
        plt.show()

    def visstd(a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5

    # Example visualization (no visual effect logic here)
    showarray(visstd(img_noise / 255.0))
    
    # Add input image processing
    input_image_path = "input.jpg"  # Place your input image in the project directory
    if not os.path.exists(input_image_path):
        raise FileNotFoundError("Please add an input image named 'input.jpg' to the project directory")
    
    # Load and preprocess input image
    input_tensor = load_image(input_image_path)
    
    # Apply visual effect
    effect_img = visual_effect(effect_model, input_tensor, num_iterations=100, lr=0.01)
    
    # Convert result to numpy array for visualization
    final_img = effect_img.squeeze().permute(1, 2, 0).numpy()
    
    # Save and display result
    plt.figure(figsize=(10, 10))
    plt.imshow(visstd(final_img))
    plt.axis('off')
    plt.savefig('output_visual_effect.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == '__main__':
    main()
