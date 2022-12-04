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

class FeatureDataset(Dataset):
	def __init__(self, data_path, col_names, normalization):
		# read csv file
		df = pd.read_csv(data_path, sep = ",", header=None, low_memory=False)
		df.columns = col_names

		y = df[['class']]
		del df['class']
		x = df

		# converting to torch tensors
		self.X_train = torch.tensor(x.values).float()
		self.y_train = torch.tensor(y.values).float()

		print(self.X_train)
		print(self.y_train)

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

def train_one_epoch(model, data_loader, loss_function, optimizer, device, losses):
	epoch_loss = 0.0
	for idx, (inputs, targets) in enumerate(data_loader):
		inputs = inputs.to(device) 
		targets = targets.to(device)
		optimizer.zero_grad()
		# calculate loss
		predictions = model(inputs)
		loss = loss_function(predictions, targets)

		# backpropagate loss and update weights
		#optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss += predictions.shape[0] * loss.item()

	cur_loss = epoch_loss/len(data_loader)
	print(f"Loss: {cur_loss}")
	losses.append(cur_loss)
	#print(f"Loss: {loss.item()}")
	#losses.append(loss.item())

def train(model, data_loader, loss_function, optimizer, device, epochs):
	writer = SummaryWriter("runs") # Visiualize Training Data
	losses = []
	for epoch in range(epochs):
		print(f"Epoch {epoch+1}")
		train_one_epoch(model, data_loader, loss_function, optimizer, device, losses)
		print("---------------------")
		writer.add_histogram("Layer 1 Weights", model.fc1.weight, epoch)
		writer.add_histogram("Layer 1 Bias", model.fc1.bias, epoch)
		writer.add_histogram("Layer 2 Weights", model.fc2.weight, epoch)
		writer.add_histogram("Layer 2 Bias", model.fc2.bias, epoch)
		writer.add_histogram("Layer 3 Weights", model.fc3.weight, epoch)
		writer.add_histogram("Layer 3 Bias", model.fc3.bias, epoch)

		writer.add_scalar("Loss/Epochs", losses[epoch], epoch)
	
	print("Training is done.")
	writer.close()
	return losses
