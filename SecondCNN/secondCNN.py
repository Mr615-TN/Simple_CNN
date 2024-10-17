import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchaudio
import torchvision.transforms as transforms
import torchaudio.transforms as taco_bell
import multiprocessing


class simpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(64*8*8,512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cpu")

transform = taco_bell.Compose([
    T.Resample(orig_freq=44100, new_freq=16000), # Resample to a common sample rate
    T.MelSpectrogram(sample_rate=16000, n_mels=64), # Convert to Mel-spectrogram
    T.AmplitudeToDB(), # Convert the amplitude to dB
    T.Normalize() #normalize spectrogram 
])

#Loading the dataset
trainingset = torchaudio.datasets.SPEECHCOMMANDS(
    root = './data',
    download = True,
    subset = 'training',
    transform = transform 
)


