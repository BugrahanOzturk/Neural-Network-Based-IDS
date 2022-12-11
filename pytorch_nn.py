# author    : Bugrahan OZTURK
# date      : 27.11.2022
# project   : Neural Networks Based Wireless Intrusion Detection System

### IMPORTS ###
import csv
import torch
import pandas as pd
import numpy as np
import os
import config
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint

#*******************************************************************#
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
#*******************************************************************#

#*******************************************************************#
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, input_num, hidden_num1, hidden_num2, hidden_num3, output_num):
        super(ShallowNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_num1)
        self.fc2 = nn.Linear(hidden_num1, hidden_num2)
        self.fc3 = nn.Linear(hidden_num2, hidden_num3)
        self.fc4 = nn.Linear(hidden_num3, output_num)
        self.apply(self._init_weights)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.my_device = torch.device('cpu') #Default to cpu
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #module.weight.data.normal_(mean=0.0, std=1.0)
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x
#*******************************************************************#

#*******************************************************************#
def train_one_epoch(model, train_data_loader, loss_function, optimizer, train_losses):
    '''
        This function is used to train the given model for 1 epoch.
    '''
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
#*******************************************************************#

#*******************************************************************#
def train(model, train_data_loader, valid_data_loader, loss_function, optimizer, epochs, val_per_epoch, validation=True, tensorboard_report=True):
    '''
        This function is used to train a given neural network model.
    '''
    if tensorboard_report:
        writer = SummaryWriter("runs") # Visiualize Training Data
    
    train_losses = []
    valid_losses = []
    accuracies = []
    validation_cnt = 0
    min_valid_loss = np.inf
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_one_epoch(model, train_data_loader, loss_function, optimizer, train_losses)
        
        if tensorboard_report:
            writer.add_histogram("Layer 1 Weights", model.fc1.weight, epoch)
            writer.add_histogram("Layer 1 Bias", model.fc1.bias, epoch)
            writer.add_histogram("Layer 2 Weights", model.fc2.weight, epoch)
            writer.add_histogram("Layer 2 Bias", model.fc2.bias, epoch)
            writer.add_histogram("Layer 3 Weights", model.fc3.weight, epoch)
            writer.add_histogram("Layer 3 Bias", model.fc3.bias, epoch)
            writer.add_histogram("Layer 4 Weights", model.fc4.weight, epoch)
            writer.add_histogram("Layer 4 Bias", model.fc4.bias, epoch)

            writer.add_scalar("Training_Loss/Epochs", train_losses[epoch], epoch)
        
        if validation and epoch%val_per_epoch == 0 and epoch != 0:
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
            
            if tensorboard_report:
                writer.add_scalar("Validation_Loss/Tries", valid_losses[validation_cnt], validation_cnt)
                writer.add_scalar("Validation_Accuracy/Tries", accuracies[validation_cnt], validation_cnt)
            
            validation_cnt += 1
        print("---------------------")    

    print("Training is done.")
    writer.close()
#*******************************************************************#

#*******************************************************************#
def validation_check(model, test_dataloader, loss_function):
    '''
        This function is used for conducting validation checks
        during the training phase.
    '''
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

            output = torch.round(output, decimals=3)
            
            # Map the output tensor
            for i in output:
                if i.item() >= 0 and i.item() < 0.25:
                    i[0] = 0.125
                elif i.item() >= 0.25 and i.item() < 0.50:
                    i[0] = 0.375
                elif i.item() >= 0.50 and i.item() < 0.75:
                    i[0] = 0.625
                elif i.item() >= 0.75 and i.item() <= 1.0:
                    i[0] = 0.875
            
            batch_acc = metric(output, targets)
            #print(f"Accuracy on batch {idx}: {batch_acc}")
    
    test_loss = test_loss/len(test_dataloader.sampler)
    accuracy = 100*metric.compute()
    metric.reset()
    return test_loss, accuracy    
#*******************************************************************#

#*******************************************************************#
def test_model(model, test_dataloader, loss_function):
    '''
        This function tests a given model in terms of loss and
        accuracy.
    '''
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

            output = torch.round(output, decimals=3)
            
            # Map the output tensor
            for i in output:
                if i.item() >= 0 and i.item() < 0.25:
                    i[0] = 0.125
                elif i.item() >= 0.25 and i.item() < 0.50:
                    i[0] = 0.375
                elif i.item() >= 0.50 and i.item() < 0.75:
                    i[0] = 0.625
                elif i.item() >= 0.75 and i.item() <= 1.0:
                    i[0] = 0.875
                y_pred.append(i.item())

            for i in targets:
                y_true.append(i.item())
            
            batch_acc = metric(output, targets)
    
    test_loss = test_loss/len(test_dataloader.sampler)
    accuracy = 100*metric.compute()
    metric.reset()
    return test_loss, accuracy, y_pred, y_true
#*******************************************************************#

#*******************************************************************#
def hyperparameter_optimizer(my_config, train_data_loader, test_dataloader):
    '''
        This function is used to tune the hyperparameters (hidden layer neurons, learning rate)
        of the given neural network model
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Construct the network
    model = ShallowNeuralNetwork(config.N_INPUTS, my_config['hidden1'], my_config['hidden2'], my_config['hidden3'], config.N_OUTPUTS).to(device)
    #model.my_device = device

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=my_config['lr'])

    # Restore checkpoint
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    train_losses = []
    y_pred = []
    y_true = []
	# train model
    for epoch in range(my_config['epochs']):
        #print(f"Epoch {epoch}")
        train_one_epoch(model, ray.get(train_data_loader), loss_fn, optimizer, train_losses)
    
        test_loss, accuracy, y_pred, y_true = test_model(model, ray.get(test_dataloader), loss_fn)    

        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "my_model")
        os.makedirs(path, exist_ok=True)
        torch.save(
            (model.state_dict(), optimizer.state_dict()), os.path.join(path, "checkpoint.pt")
        )
        checkpoint = Checkpoint.from_directory(path)
        session.report({"loss": test_loss, "accuracy": accuracy}, checkpoint=checkpoint)
    
    print("Finished training")

#*******************************************************************#

#*******************************************************************#
def plot_confusion_mtrx(y_true, y_pred):
    '''
        This function returns the confusion matrix dataframe for
        the given targets and predictions lists.
    '''
    # Map Every Floating point to a integer
    for idx, value in enumerate(y_pred):
        if value == 0.125:
            y_pred[idx] = 0
        elif value == 0.375:
            y_pred[idx] = 1
        elif value == 0.625:
            y_pred[idx] = 2
        elif value == 0.875:
            y_pred[idx] = 3
    
    for idx, value in enumerate(y_true):
        if value == 0.125:
            y_true[idx] = 0
        elif value == 0.375:
            y_true[idx] = 1
        elif value == 0.625:
            y_true[idx] = 2
        elif value == 0.875:
            y_true[idx] = 3

    cf_matrix = confusion_matrix(y_true, y_pred)
    class_names = ('flooding', 'impersonation', 'injection', 'normal')
    
    return pd.DataFrame(cf_matrix, index=[i for i in class_names], columns=[i for i in class_names])
#*******************************************************************#