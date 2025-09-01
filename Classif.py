import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

# ✅ Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Transform
transform = transforms.ToTensor()

# Datasets
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Loaders
loaders = {
    'train': torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
    'test': torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=1),
}

# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 36, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(36, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x

cnn = CNN().to(device)  # ✅ Move model to GPU

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.01)

# Training
def train(num_epochs, cnn, loaders):
    cnn.train()
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            # ✅ Move data to GPU
            b_x = images.to(device)
            b_y = labels.to(device)

            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

train(10, cnn, loaders)

# Testing
def test():
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            images = images.to(device)  # ✅ Move to GPU
            labels = labels.to(device)
            test_output, _ = cnn(images)
            pred_y = torch.max(test_output, 1)[1]
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))

test()

# Sample predictions
sample = next(iter(loaders['test']))
imgs, lbls = sample
imgs = imgs.to(device)  # ✅ Move to GPU
test_output, _ = cnn(imgs[:100])
pred_y = torch.max(test_output, 1)[1].cpu().numpy()  # ✅ Move back to CPU for numpy
print(f'Prediction number {pred_y}')
print(f'Actual number {lbls[:100].numpy()}')