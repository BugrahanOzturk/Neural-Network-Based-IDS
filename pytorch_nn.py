# author    : Bugrahan OZTURK
# date      : 27.11.2022
# project   : Neural Networks Based Wireless Intrusion Detection System

### IMPORTS ###
import csv
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torch.utils.data import WeightedRandomSampler

class FeatureDataset(Dataset):
    def __init__(self, data_path, col_names, sample_data=False):
        # read csv file
        df = pd.read_csv(data_path, sep = ",", header=None, low_memory=False)
        df.columns = col_names

        self.sampler = None
        if sample_data:
            class_counts = df['class'].value_counts(); class_counts
            class_weights = 1/class_counts; class_weights
            sample_weights = [1/class_counts[i] for i in df['class'].values]; sample_weights[:5]
            self.sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(df), replacement=True)

        
        y = df[['class']]
        del df['class']
        x = df

        # converting to torch tensors
        self.X_train = torch.tensor(x.values).float()
        self.y_train = torch.tensor(y.values).float()

        #print(self.X_train)
        #print(self.y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

class ShallowNeuralNetwork(nn.Module):
    def __init__(self, input_num, hidden_num1, hidden_num2, hidden_num3, output_num):
        super(ShallowNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_num1)
        self.fc2 = nn.Linear(hidden_num1, hidden_num2)
        self.fc3 = nn.Linear(hidden_num2, hidden_num3)
        self.fc4 = nn.Linear(hidden_num3, output_num)
        self.tanh = nn.Tanh()
        #self.softmax = nn.Softmax(dim=0)
        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.my_device = torch.device('cpu') #Default to cpu

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.relu(self.fc4(x))
        return x

def train_one_epoch(model, train_data_loader, valid_data_loader, loss_function, optimizer, device, train_losses, valid_losses, accuracies):
    train_loss = 0.0
    model.train()
    for idx, (inputs, targets) in enumerate(train_data_loader):
        inputs = inputs.to(model.my_device) 
        targets = targets.to(model.my_device)

        # clear the gradients
        optimizer.zero_grad()

        # calculate loss
        predictions = model(inputs)
        loss = loss_function(predictions, targets)

        # backpropagate loss and update weights
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        # calculate total loss for the batch by average loss(sample) * batch size
        train_loss += loss.item() * inputs.size(0)

    # calculate average loss for the epoch by dividing total batch losses by number of batches
    train_loss = train_loss/len(train_data_loader.sampler)
    print(f"Training Loss: {train_loss}")
    train_losses.append(train_loss)

def train(model, train_data_loader, valid_data_loader, loss_function, optimizer, device, epochs, validation):
    writer = SummaryWriter("runs") # Visiualize Training Data
    train_losses = []
    valid_losses = []
    accuracies = []
    validation_cnt = 0
    min_valid_loss = np.inf
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_one_epoch(model, train_data_loader, valid_data_loader, loss_function, optimizer, device, train_losses, valid_losses, accuracies)
        writer.add_histogram("Layer 1 Weights", model.fc1.weight, epoch)
        writer.add_histogram("Layer 1 Bias", model.fc1.bias, epoch)
        writer.add_histogram("Layer 2 Weights", model.fc2.weight, epoch)
        writer.add_histogram("Layer 2 Bias", model.fc2.bias, epoch)
        writer.add_histogram("Layer 3 Weights", model.fc3.weight, epoch)
        writer.add_histogram("Layer 3 Bias", model.fc3.bias, epoch)
        writer.add_histogram("Layer 4 Weights", model.fc4.weight, epoch)
        writer.add_histogram("Layer 4 Bias", model.fc4.bias, epoch)

        writer.add_scalar("Training_Loss/Epochs", train_losses[epoch], epoch)
        
        if validation and epoch%10 == 0 and epoch != 0:
            valid_loss, accuracy = validation_check(model, valid_data_loader, loss_function)
            print(f"Validation Loss: {valid_loss}")
            valid_losses.append(valid_loss)
            print(f"Accuracy: %{accuracy}")
            accuracies.append(accuracy)
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(model.state_dict(), 'feedforwardnet.pth')
            
            writer.add_scalar("Validation_Loss/Tries", valid_losses[validation_cnt], validation_cnt)
            writer.add_scalar("Validation_Accuracy/Tries", accuracies[validation_cnt], validation_cnt)
            validation_cnt += 1
        print("---------------------")    

    print("Training is done.")
    writer.close()

def validation_check(model, test_dataloader, loss_function):
    test_loss = 0.0
    model.eval()
    metric = Accuracy(task='multiclass', num_classes = 4)
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_dataloader):
            inputs = inputs.to(model.my_device)
            targets = targets.to(model.my_device)
            
            output = model(inputs)
            loss = loss_function(output, targets)
            test_loss += loss.item()*inputs.size(0)

            output = torch.round(output)
            targets = torch.round(targets)
            
            batch_acc = metric(output, targets)
            #print(f"Accuracy on batch {idx}: {batch_acc}")
    
    test_loss = test_loss/len(test_dataloader.sampler)
    accuracy = 100*metric.compute()
    metric.reset()
    return test_loss, accuracy    

def test_model(model, test_dataloader, loss_function):
    test_loss = 0.0
    model.eval()
    y_true = []
    y_pred = []
    metric = Accuracy(task='multiclass', num_classes = 4)
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_dataloader):
            inputs = inputs.to(model.my_device)
            targets = targets.to(model.my_device)
            
            output = model(inputs)
            loss = loss_function(output, targets)
            test_loss += loss.item()*inputs.size(0)

            output = torch.round(output)
            targets = torch.round(targets)

            for i in output:
                y_pred.append(i.item())
            for i in targets:
                y_true.append(i.item())
            
            batch_acc = metric(output, targets)
    
    test_loss = test_loss/len(test_dataloader.sampler)
    accuracy = 100*metric.compute()
    metric.reset()
    return test_loss, accuracy, y_pred, y_true