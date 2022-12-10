# author    : Bugrahan OZTURK
# date      : 27.11.2022
# project   : Neural Networks Based Wireless Intrusion Detection System

### IMPORTS ###
import torch
import pandas as pd
import os
import torch.nn as nn
from pytorch_nn import ShallowNeuralNetwork
from pytorch_nn import FeatureDataset
from pytorch_nn import train
import config

if __name__ == "__main__":

	dirname = os.path.dirname(__file__)
	train_file = os.path.join(dirname, "../PREPROCESSED_DATA/train_data")
	col_file = os.path.join(dirname, "../PREPROCESSED_DATA/train_columns.txt")
	test_file = os.path.join(dirname, "../PREPROCESSED_DATA/test_data")

	column_names = []

	with open(col_file) as file:
		for line in file:
			column_names.append(line.strip())

	training_data = FeatureDataset(train_file, column_names, sample_data = True)
	train_dataloader = torch.utils.data.DataLoader(training_data, sampler=training_data.sampler, batch_size = config.BATCH_SIZE_TRAIN, shuffle = False, num_workers=2) #Batch Size is set to 1 for pattern learning

	valid_data = FeatureDataset(test_file, column_names, sample_data = False)
	valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size = config.BATCH_SIZE_TEST, shuffle = False, num_workers=2)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

	# Construct the network
	feed_forward_net = ShallowNeuralNetwork(config.N_INPUTS, config.N_HIDDEN1, config.N_HIDDEN2, config.N_HIDDEN3, config.N_OUTPUTS).to(device)
	feed_forward_net.my_device = device

	loss_fn = nn.MSELoss()
	#loss_fn = nn.BCELoss()
	#optimizer = torch.optim.SGD(feed_forward_net.parameters(), lr = config.LEARNING_RATE)
	optimizer = torch.optim.Adam(feed_forward_net.parameters(), lr = config.LEARNING_RATE)

	# train model
	train(feed_forward_net, train_dataloader, valid_dataloader, loss_fn, optimizer, device, config.EPOCHS, validation=True)
         