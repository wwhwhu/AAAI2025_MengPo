import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config.config import Config

from util import separate_data, split_data, save_file, visualize_client_data_distribution_1

config = Config()
random.seed(1)
np.random.seed(1)
n_clients = config.num_clients
partition = config.partition
d_alpha = config.D_alpha
path = f"data_dir/{d_alpha}_{n_clients}/MNIST/"
non_iid = config.non_iid
balance = config.balance


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, par_algo):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset.data), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    for _, train_data in enumerate(train_loader, 0):
        train_dataset.data, train_dataset.targets = train_data
    for _, test_data in enumerate(test_loader, 0):
        test_dataset.data, test_dataset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(train_dataset.data.cpu().detach().numpy())
    dataset_image.extend(test_dataset.data.cpu().detach().numpy())
    dataset_label.extend(train_dataset.targets.cpu().detach().numpy())
    dataset_label.extend(test_dataset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, niid, balance, par_algo,
                                    class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, statistic)
    visualize_client_data_distribution_1(train_data, num_clients, num_classes,
                                       output_file=dir_path + "train_distribution.pdf")


if __name__ == "__main__":
    generate_dataset(path, n_clients, non_iid, balance, partition)
