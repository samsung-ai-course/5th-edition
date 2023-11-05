# -*- coding: utf-8 -*-
"""MNIST MLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xuNAqGST5gvH1PoXAuSmcGzqDa5bhvHU
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

"""### Cluster MNIST classes

1. Extract features using a auto-encoder
2. Reduce to 2 dimensions (TSNE or PCA)
3. Plot

1. Load and prepare data
"""

transform = transforms.Compose([transforms.ToTensor()]) # https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html -> Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8

batch_size = 64
#https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""2. Create the model"""

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_autoencoder(self, train_loader, num_epochs, device):
        self.to(device)
        for epoch in range(num_epochs):
            for data in train_loader:
                inputs, _ = data
                inputs = inputs.view(inputs.size(0), -1).to(device) # What is this doing ? #2D -> 1D for each image, N * 2D - (N * 100 * 100) -> (N * 10000)
                self.optimizer.zero_grad()
                outputs = self(inputs) #-> model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def encode_images(self, data_loader, device):
        encoded_images = []
        self.eval()
        with torch.no_grad():
            for data in data_loader:
                inputs, _ = data
                inputs = inputs.view(inputs.size(0), -1).to(device)
                encoded = self.encoder(inputs)
                encoded_images.append(encoded)

        return torch.cat(encoded_images).cpu().numpy()

"""3. Train the model"""

model = Autoencoder()
model.train_autoencoder(train_loader, 10, device)

"""4. Plot data"""

# Encode the test images
encoded_images = model.encode_images(test_loader, device)


from sklearn.decomposition import PCA

# Create a PCA instance
pca = PCA(n_components=2)

# Fit and transform the encoded_images
pca_results = pca.fit_transform(encoded_images)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(pca_results[:, 0], pca_results[:, 1], c=test_dataset.targets, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(0, 9)
plt.title("PCA Visualization of Autoencoder Encoded Features")
plt.show()

tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(encoded_images)

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=test_dataset.targets, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(ticks=range(10))
plt.clim(0, 9)
plt.title("t-SNE Visualization of Autoencoder Encoded Features")
plt.show()

"""If we try to use a MLP to classify MNIST, how good/bad would it work ?

Think about it, and then build it.

1. Define model.
2. Train
3. Evaluate
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc(x)
        return x


input_size = 28 * 28
hidden_size = 128
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 15

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = MLPModel(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss() #'multi-class loss'
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


all_labels = []
all_predictions = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Calculate precision, recall, and show the confusion matrix
from sklearn.metrics import precision_score, recall_score

precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')

print(f'Weighted Precision: {precision:.2f}')
print(f'Weighted Recall: {recall:.2f}')

# Create a confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()