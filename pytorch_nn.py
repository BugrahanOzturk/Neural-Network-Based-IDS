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

to_int16=['"radiotap.present.reserved"',
            '"wlan.fc.type_subtype"',
            '"wlan.fc.ds"',
            '"wlan_mgt.fixed.capabilities.cfpoll.ap"',
           	'"wlan_mgt.fixed.listen_ival"',
            '"wlan_mgt.fixed.status_code"', 
            '"wlan_mgt.fixed.timestamp"', 
            '"wlan_mgt.fixed.aid"', 
            '"wlan_mgt.fixed.reason_code"',
            '"wlan_mgt.fixed.auth_seq"',
            '"wlan_mgt.fixed.htact"', 
            '"wlan_mgt.fixed.chanwidth"',
            '"wlan_mgt.tim.bmapctl.offset"',
            '"wlan_mgt.country_info.environment"',
            '"wlan_mgt.rsn.capabilities.ptksa_replay_counter"', 
            '"wlan_mgt.rsn.capabilities.gtksa_replay_counter"',
			'"wlan.qos.ack"' ]

class FeatureDataset(Dataset):
	def __init__(self, data_path, col_names, normalization):
		# read csv file
		df = pd.read_csv(data_path, sep = ",", low_memory=False)

		print(df.shape)

		# Following steps at https://github.com/Bee-Mar/AWID-Intrusion-Detection
		df = df.replace('?', np.nan)
		# If over %90 of the values in a column is null, remove it
		prev_num_cols = len(df.columns)
		df.dropna(axis='columns', thresh=len(df.index)*0.10, inplace=True)
		print("Removed " + str(prev_num_cols - len(df.columns)) + " columns with all NaN values.")

		print(df.shape)

		# Drop the columns that have over %100 of its values as constant
		cols_to_drop = []
		for col in df:
			if df[col].nunique() >= (len(df.index)):
				cols_to_drop.append(col)
		df.drop(columns=cols_to_drop, inplace=True)

		print(df.shape)

		encoder = LabelEncoder()
		for column in df:
			if column in to_int16:
				df[column] = df[column].apply(lambda x: int(str(x), base=16) if x != np.nan else x)
			if column == '"class"':
				df[column] = encoder.fit_transform(df[column])

		print(encoder.classes)
		print(df.describe())

		# Normalization
		if normalization:
			df = (df-df.min())/(df.max()-df.min())

		# Set Target Class
		#y = df[col_names[-1]]
		#del df[col_names[-1]]

		# Set Features
		#x = df

		# converting to torch tensors
		#self.X_train = torch.from_numpy(x.values).float()
		#self.y_train = torch.from_numpy(y.values).float()

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
