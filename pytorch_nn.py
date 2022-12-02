# author    : Bugrahan OZTURK
# date      : 27.11.2022
# project   : Neural Networks Based Wireless Intrusion Detection System

### IMPORTS ###
import csv
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter

class FeatureDataset(Dataset):
	def __init__(self, data_path, col_names, normalization):
		# read csv file
		df = pd.read_csv(data_path, sep = ",", header=None, low_memory=False)
		df.columns = col_names
		# DATA PREPROCESSING
		# Replacing ? marks with None to find out how many ? marks are there in dataset
		df.replace({"?": None}, inplace=True)
		
		# Dropping columns with %50 of null data
		null_column = df.columns[df.isnull().mean() >= 0.6]
		df.drop(null_column, axis=1, inplace=True)
		print("Removed " + str(len(null_column)) + " columns with all NaN values.")
		
		# Drops rows with null data
		df.dropna(inplace=True)

		# Converting all columns to numeric value
		for col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='ignore')
		print(df.select_dtypes(['number']).head())

		# Drop the columns that have %100 of its values as constant
		x, y = df.select_dtypes(['number']), df['class']
		constant_col = x.columns[x.mean() == x.max()]
		x.drop(constant_col, axis=1, inplace=True)
		print("Removed " + str(len(constant_col)) + " columns with all constant values")

		# Traning Data Class Encoding
		encoder = LabelEncoder()
		y = encoder.fit_transform(y)
		print(encoder.classes_)

		# Normalization
		if normalization:
			sc = StandardScaler()
			sc.fit(x)
			scaled_x = sc.transform(x)
		#print(x.describe())
		#print(x.dtypes)

		# converting to torch tensors
		self.X_train = torch.from_numpy(scaled_x).float()
		self.y_train = torch.from_numpy(y).float()

		print(self.X_train)
		print(self.y_train)

	def __len__(self):
		return len(self.y_train)

	def __getitem__(self, idx):
		return self.X_train[idx], self.y_train[idx]

class ShallowNeuralNetwork(nn.Module):
	def __init__(self, input_num, hidden_num, output_num):
		super(ShallowNeuralNetwork, self).__init__()
		self.fc1 = nn.Linear(input_num, hidden_num)
		self.tanh = nn.Tanh()
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.fc2 = nn.Linear(hidden_num, output_num)

	def forward(self, x):
		x = self.fc1(x)
		x = self.tanh(x)
		x = self.fc2(x)
		x = self.tanh(x)
		return x

def train_one_epoch(model, data_loader, loss_function, optimizer, device, losses):
	for idx, (inputs, targets) in enumerate(data_loader):
		inputs = inputs.to(device) 
		targets = targets.to(device)
		optimizer.zero_grad()
		# calculate loss
		predictions = model(inputs)
		predictions = predictions.squeeze(-1)
		loss = loss_function(predictions, targets)

		# backpropagate loss and update weights
		#optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	print(f"Loss: {loss.item()}")
	losses.append(loss.item())

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
	
	print("Training is done.")
	writer.close()
	return losses
