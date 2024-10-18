import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchaudio
import torchaudio.transforms as taco_bell
from torchvision.transforms import Compose
import multiprocessing


class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
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
    
class Normalize(torch.nn.Module):
    def __init__(self, mean=0, std=1):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / self.std
device = torch.device("cpu")

transform = Compose([
    taco_bell.Resample(orig_freq=44100, new_freq=16000), # Resample to a common sample rate
    taco_bell.MelSpectrogram(sample_rate=16000, n_mels=64), # Convert to Mel-spectrogram
    taco_bell.AmplitudeToDB(), # Convert the amplitude to dB
    Normalize() #normalize spectrogram
])
#Loading the dataset
trainingset = torchaudio.datasets.SPEECHCOMMANDS(root = './data',download = True,subset = 'training',transform = transform)

trainloader = torch.utils.data.DataLoader(trainingset,batchsize = 32,shuffle=True,num_workers = 2)

model = simpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0
print('Finished Training')

testset = torchaudio.datasets.SPEECHCOMMANDS(root = './data',download = True,subset = 'testing',transform = transform)
testloader = torch.utils.data.DataLoader(testset,batchsize = 32,shuffle=False,num_workers = 2)

with torch.no_grad():
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct/total:.2f}%')

torch.save(model.state_dict(), 'simple_cnn.pth')
print("model saved to simple_cnn.pth")

#def main():


