import copy
import os

import numpy as np
import torch
from torch import nn


class Client(object):
    def __init__(self, args, cid):
        torch.manual_seed(0)
        self.D_alpha = args.D_alpha
        self.model = copy.deepcopy(args.model)
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.client_id = cid
        self.local_round = args.local_round
        self.learning_rate = args.lr
        self.loss = nn.CrossEntropyLoss()
        self.device = args.device
        self.num_classes = args.num_classes
        self.device = args.device
        self.client_num = args.num_clients
        self.train_data = self.load_train_data()
        self.train_samples = len(self.train_data)
        self.test_data = self.load_test_data()
        self.test_samples = len(self.test_data)

    def load_train_data(self):
        # 构建训练数据文件路径
        train_data_dir = os.path.join('data_dir', str(self.D_alpha) + f"_{self.client_num}", self.dataset, 'train')
        train_file = os.path.join(train_data_dir, f'{self.client_id}.npz')
        try:
           # 打开并加载训练数据文件
            with open(train_file, 'rb') as f:
                train_data = np.load(f, allow_pickle=True)['data'].tolist()
                X_train = torch.tensor(train_data['x'], dtype=torch.float32)
                y_train = torch.tensor(train_data['y'], dtype=torch.int64)
            # 生成训练数据列表
            train_data = [(x, y) for x, y in zip(X_train, y_train)]
            return train_data
        except FileNotFoundError:
            print(f"Error: The file {train_file} was not found.")
            return None
        except Exception as e:
            print(f"Error: An error occurred while loading the file {train_file}: {e}")
            return None

    def load_test_data(self):
        # 构建测试数据文件路径
        test_data_dir = os.path.join('data_dir', str(self.D_alpha) + f"_{self.client_num}", self.dataset, 'test')
        test_file = os.path.join(test_data_dir, f'{self.client_id}.npz')
        try:
            # 打开并加载测试数据文件
            with open(test_file, 'rb') as f:
                test_data = np.load(f, allow_pickle=True)['data'].tolist()
                X_test = torch.tensor(test_data['x'], dtype=torch.float32)
                y_test = torch.tensor(test_data['y'], dtype=torch.int64)
            # 生成测试数据列表
            test_data = [(x, y) for x, y in zip(X_test, y_test)]
            return test_data
        except FileNotFoundError:
            print(f"Error: The file {test_file} was not found.")
            return None
        except Exception as e:
            print(f"Error: An error occurred while loading the file {test_file}: {e}")
            return None
    
    def compute_sample_per_class(self):
        class_sample_counts = torch.zeros(self.num_classes, device=self.device)
        for _, y in self.train_data:
            class_sample_counts += torch.bincount(y, minlength=self.num_classes).to(self.device)
        total_samples = torch.sum(class_sample_counts)
        if total_samples > 0:
            class_sample_ratios = class_sample_counts / total_samples
        else:
            raise Exception(f"Error: No samples found for client {self.client_id}.")
        return class_sample_ratios
    
    
