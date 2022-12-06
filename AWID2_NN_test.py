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
from pytorch_nn import test_model
import config

if __name__ == "__main__":

    dirname = os.path.dirname(__file__)
    col_file = os.path.join(dirname, "../PREPROCESSED_DATA/train_columns.txt")
    test_file = os.path.join(dirname, "../PREPROCESSED_DATA/test_data")

    column_names = []

    with open(col_file) as file:
        for line in file:
            column_names.append(line.strip())


    test_data = FeatureDataset(test_file, column_names)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = config.BATCH_SIZE, shuffle = True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load the network
    feed_forward_net = ShallowNeuralNetwork(config.N_INPUTS, config.N_HIDDEN1, config.N_HIDDEN2, config.N_OUTPUTS).to(device)
    feed_forward_net.my_device = device
    pth_file = os.path.join(dirname, "feedforwardnet.pth")
    model.load_state_dict(pth_file['state_dict'])

    loss_fn = nn.MSELoss()

    # test model
    loss, accuracy = test_model(feed_forward_net, test_dataloader, loss_fn)
    print(f"Total Accuracy: {accuracy}")
    print(f"Test Loss: {loss}")