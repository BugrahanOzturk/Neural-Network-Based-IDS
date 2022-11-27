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
import matplotlib.pyplot as plt

if __name__ == "__main__":

	dirname = os.path.dirname(__file__)
	train_file = os.path.join(dirname, "../Dataset/")
	col_file = os.path.join(dirname, "../Column_Names.txt")

	column_names = []

	with open(col_file) as file:
		for line in file:
			column_names.append(line.strip())
	
	print(config.BATCH_SIZE)

	#training_data = FeatureDataset(train_file, column_names, True)
	#train_dataloader = torch.utils.data.DataLoader(training_data, batch_size = 1, shuffle = True) #Batch Size is set to 1 for pattern learning

	#for X, y in train_dataloader:s
	#	print(f"Shape of X [N, C, H, W]: {X.shape}")
	#	print(f"Shape of y: {y.shape} {y.dtype}")
	#	break

	#device = "cuda" if torch.cuda.is_available() else "cpu"
	#print(f"Using {device} device")

	# Construct the network
	#feed_forward_net = ShallowNeuralNetwork(N_INPUTS, N_HIDDEN, N_OUTPUTS).to(device)

	#loss_fn = nn.MSELoss()
	#optimizer = torch.optim.SGD(feed_forward_net.parameters(), lr = LEARNING_RATE)

	# train model
	#losses = train(feed_forward_net, train_dataloader, loss_fn, optimizer, device, EPOCHS)

	# save mode
	#torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
	#print("Model trained and stored at feedforwardnet.pth")

	# Outputs
	#plt.plot(losses)
	#plt.ylabel("loss")
	#plt.xlabel("epoch")
	#plt.title("Learning rate %f"%(LEARNING_RATE))
	#plt.show()