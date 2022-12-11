# author    : Bugrahan OZTURK
# date      : 27.11.2022
# project   : Neural Networks Based Wireless Intrusion Detection System

### IMPORTS ###
import torch
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import seaborn as sns
from matplotlib import pyplot as plt
from pytorch_nn import ShallowNeuralNetwork
from pytorch_nn import FeatureDataset
from pytorch_nn import test_model
from pytorch_nn import plot_confusion_mtrx
import config

if __name__ == "__main__":

    dirname = os.path.dirname(__file__)
    col_file = os.path.join(dirname, "../PREPROCESSED_DATA/train_columns.txt")
    test_file = os.path.join(dirname, "../PREPROCESSED_DATA/test_data")

    column_names = []

    with open(col_file) as file:
        for line in file:
            column_names.append(line.strip())


    test_data = FeatureDataset(test_file, column_names, sample_data = False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = config.BATCH_SIZE_TEST, shuffle = False, num_workers=config.NUM_WORKERS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load the network
    feed_forward_net = ShallowNeuralNetwork(config.N_INPUTS, config.N_HIDDEN1, config.N_HIDDEN2, config.N_HIDDEN3, config.N_OUTPUTS).to(device)
    feed_forward_net.my_device = device
    pth_file = os.path.join(dirname, "feedforwardnet.pth")
    feed_forward_net.load_state_dict(torch.load(pth_file))

    loss_fn = nn.MSELoss()

    # test model
    y_pred = []
    y_true = []
    loss, accuracy, y_pred, y_true = test_model(feed_forward_net, test_dataloader, loss_fn)
    print(f"Total Accuracy: {accuracy}")
    print(f"Test Loss: {loss}")

    dataframe = plot_confusion_mtrx(y_true, y_pred)

    # Plot the Confusion Matrix
    plt.figure(figsize=(12, 7))
    sns.heatmap(dataframe, annot=True, fmt='g')
    plt.title("Confusion Matrix")
    plt.ylabel("True Class"),
    plt.xlabel("Predicted Class")
    plt.show()
    