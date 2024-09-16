import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure PyTorch uses CPU
device = torch.device("cpu")

# Define a simple CNN for feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 128)  # Assuming input size is 32x32

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# SLAM algorithm
class SLAM:
    def __init__(self):
        self.feature_extractor = FeatureExtractor().to(device)
        self.optimizer = optim.Adam(self.feature_extractor.parameters())
        self.map = {}
        self.current_position = (0, 0)
        self.start_time = None
        self.checkpoints = []

    def extract_features(self, image):
        with torch.no_grad():
            features = self.feature_extractor(image.unsqueeze(0).unsqueeze(0).float().to(device))
        return features.squeeze().cpu().numpy()

    def update_map(self, features):
        self.map[self.current_position] = features

    def move(self, dx, dy):
        self.current_position = (self.current_position[0] + dx, self.current_position[1] + dy)
        self.checkpoints.append(self.current_position)

    def run(self, data):
        self.start_time = time.time()
        
        for i, (image, movement) in enumerate(data):
            features = self.extract_features(image)
            self.update_map(features)
            self.move(*movement)
            
            print(f"Checkpoint {i+1}: Position {self.current_position}")
        
        end_time = time.time()
        print(f"Algorithm runtime: {end_time - self.start_time:.2f} seconds")

# Example usage
def main():
    slam = SLAM()
    
    # Simulated data: (image, movement)
    # In a real scenario, you would replace this with actual sensor data
    simulated_data = [
        (torch.randn(32, 32), (1, 0)),
        (torch.randn(32, 32), (0, 1)),
        (torch.randn(32, 32), (-1, 0)),
    ]
    
    slam.run(simulated_data)

if __name__ == "__main__":
    main()