from google.colab import drive
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score
import random
import sys

np.set_printoptions(threshold=sys.maxsize)

# Mount Google Drive
drive.mount('/content/drive')

# Load the CSV files
train_file_path = '/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/gestures_b.csv'
train2_file_path = '/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/gestures.csv'
train3_file_path = '/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/gestures_1207.csv'
validation_file_path = '/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/gestures_n.csv'
train_data = pd.read_csv(train_file_path, header=None)
train2_data = pd.read_csv(train2_file_path, header=None)
train3_data = pd.read_csv(train2_file_path, header=None)
validation_data = pd.read_csv(validation_file_path, header=None)


X0 = train_data.iloc[:, 1:].values  # Features
y0 = train_data.iloc[:, 0].values   # Labels
X1 = train2_data.iloc[:, 1:].values  # Features
y1 = train2_data.iloc[:, 0].values   # Labels
X3 = train3_data.iloc[:, 1:].values  # Features
y3 = train3_data.iloc[:, 0].values   # Labels
X2 = validation_data.iloc[:, 1:].values  # Features
y2 = validation_data.iloc[:, 0].values   # Labels

X = np.concatenate((X0, X1, X2, X3), axis=0)
y = np.concatenate((y0, y1, y2, y3), axis=0)

print(X.shape)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Normalize the features (fit on training data only)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)

# Save the preprocessed data in .npy format
np.save('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/X_train_scaled.npy', X_train_scaled)
np.save('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/y_train.npy', y_train)
np.save('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/X_validation_scaled.npy', X_validation_scaled)
np.save('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/y_validation.npy', y_validation)

# Load preprocessed data
X_train = np.load('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/X_train_scaled.npy')
y_train = np.load('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/y_train.npy')
X_validation = np.load('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/X_validation_scaled.npy')
y_validation = np.load('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/y_validation.npy')

# Create a custom dataset
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# die Daten für PyTorch kompatibel zu machen.
train_dataset = GestureDataset(X_train, y_train)
validation_dataset = GestureDataset(X_validation, y_validation)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# Define a simple neural network
class GestureNet(nn.Module):
    def __init__(self):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return x

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Das Gerät (CPU oder GPU) wird konfiguriert.


# Das Modell, die Verlustfunktion, der Optimierer und die Metrik für die Genauigkeit werden initialisiert.
model = GestureNet().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

accuracy = Accuracy(task="multiclass", num_classes=7).to(device)
precision = Precision(task="multiclass", num_classes=7, average='macro').to(device)
recall = Recall(task="multiclass", num_classes=7, average='macro').to(device)
f1 = F1Score(task="multiclass", num_classes=7, average='macro').to(device)

# Lists to store metrics ************************************************************************
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []


# Training loop
num_epochs = 25

for epoch in range(num_epochs):
    model.train() # train() sets the modules in the network in training mode. It tells our model that we are currently in the training phase so the model keeps some layers, like dropout, batch-normalization which behaves differently depends on the current phase, active. whereas the model. eval() does the opposite.
    train_loss, correct_train = 0, 0
    for inputs, labels in train_loader:
        # Verschiebt die Eingaben und Labels auf das konfigurierte Gerät (CPU oder GPU).
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs) # Führt einen Vorwärtsdurchlauf durch das Modell mit den Eingabedaten.
        loss = criterion(outputs, labels) # Berechnet den Verlust zwischen den Modellvorhersagen und den tatsächlichen Labels.
        optimizer.zero_grad() # Setzt die Gradienten der Modellparameter auf Null, um eine Akkumulation von Gradienten aus vorherigen Batches zu vermeiden.
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct_train += accuracy(outputs, labels).item()

    train_loss /= len(train_loader)
    train_accuracy = correct_train / len(train_loader)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    # print(f'Epoch {epoch+1}, Train -- Loss: {correct_train:.4f}')

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss, correct_val = 0, 0
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            correct_val += accuracy(outputs, labels).item()
            precision(outputs, labels)
            recall(outputs, labels)


    val_loss /= len(validation_loader)
    val_accuracy = correct_val / len(validation_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_precisions.append(precision.compute().item())
    val_recalls.append(recall.compute().item())

    scheduler.step(train_loss)
    scheduler.step(val_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    print(f'val Precision: {precision.compute():.4f}, val Recall: {recall.compute():.4f}')


import matplotlib.pyplot as plt

# Plot the results
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(20, 4))

# Plot training and validation loss
plt.subplot(1, 4, 1)
plt.plot(epochs, train_losses, marker='o', label='Training Loss')
plt.plot(epochs, val_losses, marker='o', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Plot training and validation accuracy
plt.subplot(1, 4, 2)
plt.plot(epochs, train_accuracies, marker='o', label='Training Accuracy')
plt.plot(epochs, val_accuracies, marker='o', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot correct predictions (train and val accuracy)
plt.subplot(1, 4, 3)
plt.plot(epochs, np.array(train_accuracies) * 100, marker='o', label='Correct Train Predictions (%)')
plt.plot(epochs, np.array(val_accuracies) * 100, marker='o', label='Correct Val Predictions (%)')
plt.xlabel('Epochs')
plt.ylabel('Correct Predictions (%)')
plt.grid(True)
plt.legend(loc='lower right')
plt.title('Correct Predictions')

plt.tight_layout()
plt.show()
