# author    : Bugrahan OZTURK
# date      : 27.11.2022
# project   : Neural Networks Based Wireless Intrusion Detection System

### IMPORTS ###
import csv
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_metrics import Accuracy

class FeatureDataset(Dataset):
    def __init__(self, data_path, col_names):
        # read csv file
        df = pd.read_csv(data_path, sep = ",", header=None, low_memory=False)
        df.columns = col_names

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
    def __init__(self, input_num, hidden_num1, hidden_num2, output_num):
        super(ShallowNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_num1)
        self.fc2 = nn.Linear(hidden_num1, hidden_num2)
        self.fc3 = nn.Linear(hidden_num2, output_num)
        self.tanh = nn.Tanh()
        #self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

def train_one_epoch(model, train_data_loader, valid_data_loader, loss_function, optimizer, device, train_losses, valid_losses, accuracies):
    train_loss = 0.0
    model.train()
    for idx, (inputs, targets) in enumerate(train_data_loader):
        inputs = inputs.to(device) 
        targets = targets.to(device)

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
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_one_epoch(model, train_data_loader, valid_data_loader, loss_function, optimizer, device, train_losses, valid_losses, accuracies)
        writer.add_histogram("Layer 1 Weights", model.fc1.weight, epoch)
        writer.add_histogram("Layer 1 Bias", model.fc1.bias, epoch)
        writer.add_histogram("Layer 2 Weights", model.fc2.weight, epoch)
        writer.add_histogram("Layer 2 Bias", model.fc2.bias, epoch)
        writer.add_histogram("Layer 3 Weights", model.fc3.weight, epoch)
        writer.add_histogram("Layer 3 Bias", model.fc3.bias, epoch)

        writer.add_scalar("Training_Loss/Epochs", train_losses[epoch], epoch)
        
        if validation and epoch%10 == 0 and epoch != 0:
            valid_loss, accuracy = check_eval(model, valid_data_loader, loss_function)
            print(f"Validation Loss: {valid_loss}")
            valid_losses.append(valid_loss)
            print(f"Accuracy: %{accuracy}")
            accuracies.append(accuracy)

            writer.add_scalar("Validation_Loss/Tries", valid_losses[validation_cnt], validation_cnt)
            writer.add_scalar("Validation_Accuracy/Tries", accuracies[validation_cnt], validation_cnt)
            validation_cnt += 1
        print("---------------------")    

    print("Training is done.")
    writer.close()

def check_eval(model, valid_data_loader, loss_function):
    valid_loss = 0.0
    correct_preds = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(valid_data_loader):
            
            #forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            output = torch.round(output, decimals=2)
            # calculate the loss
            loss = loss_function(output, targets)
            # update running validation loss
            #output = torch.round(output, decimals=2)
            valid_loss += loss.item()*inputs.size(0)
            correct_preds += (output == targets).sum().item()
    
    valid_loss = valid_loss/len(valid_data_loader.sampler)
    accuracy = (100 * correct_preds/len(valid_data_loader.dataset))
    return valid_loss, accuracy

def test_model(model, test_dataloader, loss_function):
    test_loss = 0.0
    accuracy = 0
    correct_preds = 0
    model.eval()
    metric = Accuracy()
    metric.reset()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_dataloader):
            output = model(inputs)
            loss = loss_function(output, targets)
            output = torch.round(output, decimals=2)
            test_loss += loss.item()*inputs.size(0)
            batch_acc = metric(output, targets)
            print(f"Accuracy on batch {idx}: {batch_acc}")
    
    test_loss = test_loss/len(test_dataloader.sampler)
    accuracy = metric.compute()
    return test_loss, accuracy