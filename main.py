import argparse
import copy
import os

import torch
from FL_core.serverMP import MP
from config.config import Config
from model.model import FeatureHeadSplit
from util.util import return_model_args
import torch.nn as nn

def run(args):
    args.retrain = False
    args = return_model_args(args)  # 返回转化后的模型
    print("model: ", args.model)
    print("device: ", args.device)
    algorithm = args.algorithm
    if algorithm == "MP":     
        args.head = copy.deepcopy(args.model.fc)
        # args.head是一层fully connected层，将其input shape变为原来两倍，output不变
        in_features = args.head.in_features
        out_features = args.head.out_features
        args.head = nn.Linear(in_features * 2, out_features)
        args.model.fc = nn.Identity()
        args.model = FeatureHeadSplit(args.model, args.head)
        print("model: ", args.model)
        server = MP(args)
        server.train()
        print("Finish training")
    else:
        raise NotImplementedError

if __name__ == "__main__":
    config = Config()
    if config.dataset == "MNIST":
        cdevice = 2
    elif config.dataset == "FMNIST":
        cdevice = 1
    elif config.dataset == "CIFAR10":
        cdevice = 2
    elif config.dataset == "SVHN":
        cdevice = 3
    else:
        cdevice = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', "--dataset", type=str, default=config.dataset)
    parser.add_argument('-model', "--model", type=str, default=config.backbone)
    parser.add_argument('-num_classes', "--num_classes", type=int, default=config.num_classes)
    parser.add_argument('-num_clients', "--num_clients", type=int, default=config.num_clients)
    parser.add_argument('-ratio', "--ratio", type=float, default=1.0)
    parser.add_argument('-batch_size', "--batch_size", type=int, default=config.batch_size)
    parser.add_argument('-server_round', "--server_round", type=int, default=config.server_round)
    parser.add_argument('-local_round', "--local_round", type=int, default=config.local_round)
    parser.add_argument('-lr', "--lr", type=float, default=config.learning_rate)
    parser.add_argument('-eval_every', "--eval_every", type=int, default=config.eval_every)
    parser.add_argument('-algorithm', "--algorithm", type=str, default="MP")
    parser.add_argument('-lamda', "--lamda", type=int, default=0.01)
    parser.add_argument('-L2_weight', "--L2_weight", type=float, default=config.L2_weight)
    parser.add_argument('-device', "--device", type=str, default=f"cuda:{cdevice}" if torch.cuda.is_available() else "cpu")
    parser.add_argument('-head_aggregation', "--head_aggregation", type=bool, default=config.head_aggregation)
    parser.add_argument('-D_alpha', "--D_alpha", type=float, default=config.D_alpha)
    arg = parser.parse_args()
    run(arg)