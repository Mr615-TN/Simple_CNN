import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms 

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(64*8*8,512)
        self.fc2 = nn.Linear(512, 10)
    
    
