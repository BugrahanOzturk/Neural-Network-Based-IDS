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
    def __init__(self, input_num, hidden_num, output_num):
        super(ShallowNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_num)
        self.fc2 = nn.Linear(hidden_num, output_num)
        self.tanh = nn.Tanh()
        #self.softmax = nn.Softmax(dim=0)
        #self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()
        self.my_device = torch.device('cpu') #Default to cpu

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
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
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_one_epoch(model, train_data_loader, valid_data_loader, loss_function, optimizer, device, train_losses, valid_losses, accuracies)
        writer.add_histogram("Layer 1 Weights", model.fc1.weight, epoch)
        writer.add_histogram("Layer 1 Bias", model.fc1.bias, epoch)
        writer.add_histogram("Layer 2 Weights", model.fc2.weight, epoch)
        writer.add_histogram("Layer 2 Bias", model.fc2.bias, epoch)

        writer.add_scalar("Training_Loss/Epochs", train_losses[epoch], epoch)
        
        if validation and epoch%20 == 0 and epoch != 0:
            valid_loss, accuracy = validation_check(model, valid_data_loader, loss_function)
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
            
            # Cast Floating points to labels (integers) based on threshold
            for i in output:
                value = round(i.item(), 2)
                if value <= 0.15:
                    i[0] = 1 #"flooding"
                elif value > 0.15 and value <= 0.45:
                    i[0] = 2 #"impersonation"
                elif value > 0.45 and value <= 0.85:
                    i[0] = 3 #"injection"
                elif value > 0.85:
                    i[0] = 4 #"normal"
            
            for i in targets:
                value = round(i.item(), 1)
                if value == 0.0:
                    i[0] = 1 #"flooding"
                elif value == 0.3:
                    i[0] = 2 #"impersonation"
                elif value == 0.7:
                    i[0] = 3 #"injection"
                elif value == 1.0:
                    i[0] = 4 #"normal"
            
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
            
            # Convert Floating Points to integers for confusion matrix
            for i in output:
                value = round(i.item(), 2)
                if value <= 0.15:
                    i[0] = 1 #"flooding"
                elif value > 0.15 and value <= 0.45:
                    i[0] = 2 #"impersonation"
                elif value > 0.45 and value <= 0.85:
                    i[0] = 3 #"injection"
                elif value > 0.85:
                    i[0] = 4 #"normal"
                y_pred.append(i.item())
            
            for i in targets:
                value = round(i.item(), 1)
                if value == 0.0:
                    i[0] = 1 #"flooding"
                elif value == 0.3:
                    i[0] = 2 #"impersonation"
                elif value == 0.7:
                    i[0] = 3 #"injection"
                elif value == 1.0:
                    i[0] = 4 #"normal"
                y_true.append(i.item())
            
            batch_acc = metric(output, targets)
    
    #print(set(y_pred))
    #print(set(y_true))
    test_loss = test_loss/len(test_dataloader.sampler)
    accuracy = 100*metric.compute()
    metric.reset()
    return test_loss, accuracy, y_pred, y_true

def visualise_dataloader(dl, id_to_label=None, with_outputs=True):
    total_num_images = len(dl.dataset)
    idxs_seen = []
    class_0_batch_counts = []
    class_1_batch_counts = []
    class_2_batch_counts = []
    class_3_batch_counts = []

    for i, batch in enumerate(dl):

        idxs = batch[1][:, 0].tolist()
        classes = batch[1][:, 0]
        class_ids, class_counts = classes.unique(return_counts=True)
        class_ids = set(class_ids.tolist())
        class_counts = class_counts.tolist()

        idxs_seen.extend(idxs)

        if len(class_ids) == 2:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(class_counts[1])
        elif len(class_ids) == 1 and 0 in class_ids:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(0)
        elif len(class_ids) == 1 and 1 in class_ids:
            class_0_batch_counts.append(0)
            class_1_batch_counts.append(class_counts[0])
        elif len(class_ids) == 4:
            class_0_batch_counts.append(class_counts[0])
            class_1_batch_counts.append(class_counts[1])
            class_2_batch_counts.append(class_counts[2])
            class_3_batch_counts.append(class_counts[3])
        #else:
            #raise ValueError("More than two classes detected")

    if with_outputs:
        fig, ax = plt.subplots(1, figsize=(15, 15))

        ind = np.arange(len(class_0_batch_counts))
        width = 0.35

        ax.bar(
            ind,
            class_0_batch_counts,
            width,
            label=(id_to_label[0] if id_to_label is not None else "0"),
        )
        ax.bar(
            ind + width,
            class_1_batch_counts,
            width,
            label=(id_to_label[1] if id_to_label is not None else "1"),
        )
        ax.bar(
            ind + width,
            class_2_batch_counts,
            width,
            label=(id_to_label[2] if id_to_label is not None else "2"),
        )
        ax.bar(
            ind + width,
            class_3_batch_counts,
            width,
            label=(id_to_label[3] if id_to_label is not None else "3"),
        )
        ax.set_xticks(ind, ind + 1)
        ax.set_xlabel("Batch index", fontsize=12)
        ax.set_ylabel("No. of images in batch", fontsize=12)
        ax.set_aspect("equal")

        plt.legend()
        plt.show()

        num_images_seen = len(idxs_seen)

        print(
            f'Avg Proportion of {(id_to_label[0] if id_to_label is not None else "Class 0")} per batch: {(np.array(class_0_batch_counts) / 10).mean()}'
        )
        print(
            f'Avg Proportion of {(id_to_label[1] if id_to_label is not None else "Class 1")} per batch: {(np.array(class_1_batch_counts) / 10).mean()}'
        )
        print(
            f'Avg Proportion of {(id_to_label[2] if id_to_label is not None else "Class 2")} per batch: {(np.array(class_2_batch_counts) / 10).mean()}'
        )
        print(
            f'Avg Proportion of {(id_to_label[3] if id_to_label is not None else "Class 3")} per batch: {(np.array(class_3_batch_counts) / 10).mean()}'
        )
        print("=============")
        print(f"Num. unique samples seen: {len(set(idxs_seen))}/{total_num_images}")
    return class_0_batch_counts, class_1_batch_counts, class_2_batch_counts, class_3_batch_counts, idxs_seen