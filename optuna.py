!pip install torchmetrics
!pip install optuna

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
import optuna
from optuna.trial import TrialState

np.set_printoptions(threshold=sys.maxsize)

# Function to unmount and remount Google Drive
def remount_drive():
    drive_path = '/content/drive'
    if os.path.ismount(drive_path):
        !fusermount -u drive_path
    drive.mount(drive_path)

# Remount the drive
remount_drive()

# Load the CSV files
train_file_path_1 = '/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/gestures_b.csv'
train_file_path_2 = '/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/gestures.csv'
train_file_path_3 = '/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/gestures_1207.csv'
validation_file_path = '/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/gestures_n.csv'
train_data_1 = pd.read_csv(train_file_path_1, header=None)
train_data_2 = pd.read_csv(train_file_path_2, header=None)
train_data_3 = pd.read_csv(train_file_path_2, header=None)
validation_data = pd.read_csv(validation_file_path, header=None)


x_0 = train_data_1.iloc[:, 1:].values  # Features
y_0 = train_data_1.iloc[:, 0].values   # Labels
x_1 = train_data_2.iloc[:, 1:].values  # Features
y_1 = train_data_2.iloc[:, 0].values   # Labels
x_2 = train_data_3.iloc[:, 1:].values  # Features
y_2 = train_data_3.iloc[:, 0].values   # Labels
x_3 = validation_data.iloc[:, 1:].values  # Features
y_3 = validation_data.iloc[:, 0].values   # Labels

x = np.concatenate((x_0, x_1, x_2, x_3), axis=0)
y = np.concatenate((y_0, y_1, y_2, y_3), axis=0)

print(x.shape)
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# Normalize the features (fit on training data only)
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_validation_scaled = scaler.transform(x_validation)

# Save the preprocessed data in .npy format
np.save('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/x_train_scaled.npy', x_train_scaled)
np.save('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/y_train.npy', y_train)
np.save('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/x_validation_scaled.npy', x_validation_scaled)
np.save('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/y_validation.npy', y_validation)

# Load preprocessed data
x_train = np.load('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/x_train_scaled.npy')
y_train = np.load('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/y_train.npy')
x_validation = np.load('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/x_validation_scaled.npy')
y_validation = np.load('/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/y_validation.npy')

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
train_dataset = GestureDataset(x_train, y_train)
validation_dataset = GestureDataset(x_validation, y_validation)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

# Define a simple neural network
class GestureNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout_rate):
        super(GestureNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 7)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # remove ReLU activation from the final layer for classification tasks
        return x

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Das Gerät (CPU oder GPU) wird konfiguriert.

accuracy = Accuracy(task="multiclass", num_classes=7).to(device)
precision = Precision(task="multiclass", num_classes=7, average='macro').to(device)
recall = Recall(task="multiclass", num_classes=7, average='macro').to(device)

# Define the objective function
def objective(trial):
    hidden_size1 = trial.suggest_int('hidden_size1', 180, 260)
    hidden_size2 = trial.suggest_int('hidden_size2', 70, 120)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)
    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
    num_epochs = trial.suggest_int('epochs', 35, 55)

    model = GestureNet(x_train.shape[1], hidden_size1, hidden_size2, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_train = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct_train += accuracy(outputs, labels).item()

        train_loss /= len(train_loader)
        train_accuracy = correct_train / len(train_loader)

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

        scheduler.step(val_loss)

    return val_loss

# Create the study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=40)

# Print the best hyperparameters
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
print('Best hyperparameters: ', study.best_params)


# Train the model with the best hyperparameters
best_params = study.best_params
model = GestureNet(x_train.shape[1], best_params['hidden_size1'], best_params['hidden_size2'], best_params['dropout_rate']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

accuracy = Accuracy(task="multiclass", num_classes=7).to(device)
precision = Precision(task="multiclass", num_classes=7, average='macro').to(device)
recall = Recall(task="multiclass", num_classes=7, average='macro').to(device)
f1 = F1Score(task="multiclass", num_classes=7, average='macro').to(device)

num_epochs = best_params['epochs']

# Lists to store metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []

# Training loop

for epoch in range(num_epochs):
    model.train() # It tells our model that we are currently in the training phase so the model keeps some layers, like dropout, batch-normalization which behaves differently depends on the current phase, active. eval() does opposite.
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
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, val Precision: {precision.compute():.4f}, val Recall: {recall.compute():.4f}')

# Save the trained model
model_path = '/content/drive/MyDrive/Uni/SMT_Colab/Final_Projekt/results/gesture_model.pth'
torch.save(model.state_dict(), model_path)

