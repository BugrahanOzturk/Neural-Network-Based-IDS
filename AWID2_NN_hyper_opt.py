# author    : Bugrahan OZTURK
# date      : 27.11.2022
# project   : Neural Networks Based Wireless Intrusion Detection System

### IMPORTS ###
import numpy as np
import os
import torch
import config
from pytorch_nn import hyperparameter_optimizer
from pytorch_nn import FeatureDataset
from pytorch_nn import ShallowNeuralNetwork
import torch.nn as nn
from pytorch_nn import test_model
from pytorch_nn import plot_confusion_mtrx

import ray
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray import air

num_samples = 1
max_num_epochs = 1

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
    train_dataloader = torch.utils.data.DataLoader(training_data, sampler=training_data.sampler, batch_size = config.BATCH_SIZE_TRAIN, shuffle = False, num_workers=config.NUM_WORKERS) #Batch Size is set to 1 for pattern learning

    test_data = FeatureDataset(test_file, column_names, sample_data = False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = config.BATCH_SIZE_TEST, shuffle = False, num_workers=config.NUM_WORKERS)
    
    # To avoid implicit function size too large error, we pass the ids of
    # loaders to the implicit actor function (pointer of the loader instead of object)
    train_id = ray.put(train_dataloader)
    test_id = ray.put(test_dataloader)

    #search_space = {
    #    "epochs": 15,
    #    "lr": tune.loguniform(1e-4, 1e-1),
    #    "hidden1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    #    "hidden2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    #    "hidden3": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    #}

    search_space = {
        "epochs": 1,
        "lr": 0.01,
        "hidden1": tune.sample_from(lambda _: 2**np.random.randint(2, 5)),
        "hidden2": tune.sample_from(lambda _: 2**np.random.randint(2, 5)),
        "hidden3": tune.sample_from(lambda _: 2**np.random.randint(2, 5)),
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(partial(hyperparameter_optimizer, train_data_loader=train_id, test_dataloader=test_id)),
            resources={"cpu":2, "gpu":0}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples
        ),
        param_space=search_space,
        run_config=air.RunConfig(
            name="test_experiment",
            local_dir="./tune_results"
        ),
    )

    results = tuner.fit()

    logdir = results.get_best_result("loss", mode="min").log_dir
    print(results.get_best_result("loss", mode="min").checkpoint)
    state_dict = torch.load(os.path.join(logdir, "checkpoint.pt"))

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final test loss: {}".format(
        best_result.metrics["loss"]
    ))
    print("Best trial final test accuracy: {}".format(
        best_result.metrics["accuracy"]
    ))

    ## Test the best resulting Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load the network
    feed_forward_net = ShallowNeuralNetwork(config.N_INPUTS, best_result.config['hidden1'], best_result.config['hidden2'], best_result.config['hidden3'], config.N_OUTPUTS).to(device)
    feed_forward_net.my_device = device

    feed_forward_net.load_state_dict(state_dict)

    loss_fn = nn.MSELoss()

    # test model
    y_pred = []
    y_true = []
    loss, accuracy, y_pred, y_true = test_model(feed_forward_net, test_dataloader, loss_fn)
    print(f"Total Accuracy: {accuracy}")
    print(f"Test Loss: {loss}")

    torch.save(feed_forward_net.state_dict(), 'feedforwardnet.pth')

    dataframe = plot_confusion_mtrx(y_true, y_pred)

    # Plot the Confusion Matrix
    plt.figure(figsize=(12, 7))
    sns.heatmap(dataframe, annot=True, fmt='g')
    plt.title("Confusion Matrix")
    plt.ylabel("True Class"),
    plt.xlabel("Predicted Class")
    plt.show()

