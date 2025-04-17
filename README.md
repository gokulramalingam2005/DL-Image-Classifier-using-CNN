
# DL-Convolutional Deep Neural Network for Image Classification

# AIM
# To develop a convolutional neural network (CNN) classification model for the given dataset.

# THEORY
# The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels.
# The task is to classify these images into their respective digit categories using a CNN.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# DESIGN STEPS
# STEP 1: Import necessary libraries and load the dataset
# STEP 2: Define the CNN architecture
# STEP 3: Define the loss function and optimizer
# STEP 4: Train the model on the training data
# STEP 5: Evaluate the model on the test data
# STEP 6: Predict a new sample image

# Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# Neural Network Model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=10):
    loss_per_epoch = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        loss_per_epoch.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return loss_per_epoch

# Train the model
losses = train_model(model, train_loader)

# OUTPUT
# Training Loss per Epoch
plt.plot(losses)
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()

# Evaluate the model
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, output_dict=False)
print("Classification Report:")
print(report)

# New Sample Data Prediction
sample_image, label = test_set[0]
model.eval()
with torch.no_grad():
    sample_output = model(sample_image.unsqueeze(0))
    _, predicted_label = torch.max(sample_output, 1)
    print(f"Predicted Label: {predicted_label.item()}, Actual Label: {label}")
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.title(f"Prediction: {predicted_label.item()} | Actual: {label}")
    plt.savefig("sample_prediction.png")
    plt.show()

# RESULT
# The CNN model was successfully trained and tested on the MNIST dataset.
# The training loss decreased over epochs, as shown in the training loss plot (training_loss.png).
# The confusion matrix (confusion_matrix.png) and classification report indicate good classification performance.
# A new sample image was correctly classified by the model (see sample_prediction.png).

Include your result here
