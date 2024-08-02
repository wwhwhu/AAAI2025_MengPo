import copy
import csv
import os
import time

import numpy as np
import torch
from FL_core.clientMP import clientMP
from FL_core.fedserver import Server
from model.model import Global_Embedding_Generator, HyperNetwork

class MP(Server):
    def __init__(self, args):
        super(MP, self).__init__(args)
        self.embedding_dim = list(args.model.head.parameters())[0].shape[1] // 2
        # cEm init
        args.Embedding_Generator = Global_Embedding_Generator(self.embedding_dim, args.num_classes, args.device)
        self.Embedding_Generator = copy.deepcopy(args.Embedding_Generator).to(args.device)
        # global model init 
        self.global_model = copy.deepcopy(args.model).feature_extractor.to(args.device)
        self.head = copy.deepcopy(args.model.head).to(args.device)
        self.num_layers = 0
        for name, module in self.global_model.named_modules():
            print(f"层名称：{name}")
            print(f"参数数量：{sum(p.numel() for p in module.parameters())}")
            self.num_layers += 1
        print(f"num_layers: {self.num_layers}")
        # hypernetwork init
        args.HyperNetwork = HyperNetwork(self.embedding_dim, self.num_layers).to(args.device)
        self.HyperNetwork = copy.deepcopy(args.HyperNetwork).to(args.device)
        if not args.retrain:
            self.save_dir = f'save_dir/{args.D_alpha}_{args.num_clients}'
            for i in range(args.num_clients):
                client = clientMP(args, i)
                self.clients.append(client)
            # CSV文件的路径
            self.csv_path = f'./{self.save_dir}/{self.dataset_name}/client_metrics.csv'
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            # 初始化CSV文件并写入表头
            self.init_learn_csv()
        else:
            self.save_dir = f'retrain_dir/{args.D_alpha}_{args.num_clients}'
            for i in range(args.num_clients):
                if i not in args.un_client:
                    client = clientMP(args, i)
                    self.clients.append(client)
            self.csv_path = f'./{self.save_dir}/{self.dataset_name}/client_metrics.csv'
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self.init_csv()

    def train(self):
        for i in range(1, self.server_round+1):
            print(f'\n\033[0;32;40m---- Round {i} ----\033[0m')
            self.select_clients()
            time_start = time.time()
            for client in self.selected_clients:
                # 训练全局模型
                client.train()
            # 从客户端接收特征提取模型参数
            self.update_from_clients()
            # 更新模型到client
            self.update_to_clients()
            # 从客户端接收Embedding模型更新全局
            self.update_global_embedding()
            # 更新Embedding模型到client
            self.update_global_embedding_to_clients()
            # 更新head
            self.update_head()
            # 更新head模型到client
            self.update_head_to_clients()
            # 从客户端接收Embedding模型更新全局
            self.update_global_embedding()
            # 更新Embedding模型到client
            self.update_global_embedding_to_clients()
            # 更新HyperNetwork
            self.update_hypernetwork()
            # 更新HyperNetwork到client
            self.update_hypernetwork_to_clients()
            time_cost = time.time() - time_start
            if i % self.eval_every == 0:
                print(f'Training {time_cost}s, Evaluating and Saving...')
                self.test_and_save(i, time_cost)
    
    def init_learn_csv(self):
        # 初始化CSV文件并写入表头
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 动态生成表头
            headers = ['Round']
            for client in self.clients:
                headers.append(f'Client_{client.client_id}_Test_Acc')
                headers.append(f'Client_{client.client_id}_Train_Acc')
                headers.append(f'Client_{client.client_id}_Loss')
            headers.append('time_cost')
            writer.writerow(headers)
    
    # 接收客户端的模型参数并更新全局模型
    def update_from_clients(self):
        assert len(self.selected_clients) > 0, "No selected clients to update from."
        # 计算samples的总数
        total_samples = sum(client.train_samples for client in self.selected_clients)
        # 初始化全局模型参数
        global_params = [torch.zeros_like(param) for param in
                         self.selected_clients[0].model.feature_extractor.parameters()]
        # 计算全局模型参数，加权平均，只更新feature_extractor的参数
        for client in self.selected_clients:
            for i, param in enumerate(client.model.feature_extractor.parameters()):
                global_params[i] += param.data * (client.train_samples / total_samples)
        # 更新全局feature_extractor模型参数
        for i, param in enumerate(self.global_model.parameters()):
            param.data.copy_(global_params[i])
        print("Global feature extractor model updated from clients.")

    def update_to_clients(self):
        assert len(self.selected_clients) > 0, "No selected clients to update to."
        for client in self.selected_clients:
            client.update_from_global(self.global_model)
        print("Global feature extractor model updated to clients.")

    def update_head(self):
        # 计算samples的总数
        total_samples = sum(client.train_samples for client in self.selected_clients)
        # 初始化全局模型参数
        global_params = [torch.zeros_like(param) for param in
                         self.selected_clients[0].model.head.parameters()]
        # 计算全局模型参数，加权平均，只更新head的参数
        for client in self.selected_clients:
            for i, param in enumerate(client.model.head.parameters()):
                global_params[i] += param.data * (client.train_samples / total_samples)
        # 更新全局head模型参数
        for i, param in enumerate(self.head.parameters()):
            param.data.copy_(global_params[i])
        print("Global head model updated from clients.")    
    
    def update_head_to_clients(self):
        for client in self.selected_clients:
            client.update_head_from_global(self.head)
        print("Global head model updated to clients.")
    
    def update_global_embedding(self):
        # 计算samples的总数
        total_samples = sum(client.train_samples for client in self.selected_clients)
        # 初始化全局Global_Embedding_Generator模型参数
        global_params = [torch.zeros_like(param) for param in
                         self.selected_clients[0].Global_Embedding_Generator.parameters()]
        # 计算Global_Embedding_Generator模型参数，加权平均
        for client in self.selected_clients:
            for i, param in enumerate(client.Global_Embedding_Generator.parameters()):
                global_params[i] += param.data * (client.train_samples / total_samples)
        # 更新全局模型参数
        for i, param in enumerate(self.Embedding_Generator.parameters()):
            param.data.copy_(global_params[i])
        print("Global embedding_generator updated from clients.")

    def update_global_embedding_to_clients(self):
        for client in self.selected_clients:
            client.update_from_global_embedding(self.Embedding_Generator)
        print("Global embedding_generator updated to clients.")
    
    def update_hypernetwork(self):
        # 计算samples的总数
        total_samples = sum(client.train_samples for client in self.selected_clients)
        # 初始化全局HyperNetwork模型参数
        global_params = [torch.zeros_like(param) for param in
                         self.selected_clients[0].HyperNetwork.parameters()]
        # 计算HyperNetwork模型参数，加权平均
        for client in self.selected_clients:
            for i, param in enumerate(client.HyperNetwork.parameters()):
                global_params[i] += param.data * (client.train_samples / total_samples)
        # 更新全局模型参数
        for i, param in enumerate(self.HyperNetwork.parameters()):
            param.data.copy_(global_params[i])
        print("Global hypernetwork updated from clients.")
    
    def update_hypernetwork_to_clients(self):
        for client in self.selected_clients:
            client.update_hypernetwork_from_global(self.HyperNetwork)
        print("Global hypernetwork updated to clients.")

    def test_and_save(self, round_num, time_cost):
        num_samples_list = []
        total_correct_list = []
        num_samples_list2 = []
        total_correct_list2 = []
        metrics = [round_num]  # 保存当前轮次和每个客户端的度量信息
        losses = []
        for i,client in enumerate(self.clients):
            correct, samples = client.test(if_test=True)
            total_correct_list.append(correct)
            num_samples_list.append(samples)
            correct2, samples2, loss = client.test(if_test=False)
            total_correct_list2.append(correct2)
            num_samples_list2.append(samples2)
            losses.append(loss * 1.0)
            print(f'Client {client.client_id}: Train loss: {loss * 1.0 / samples2}')
            print(f'Client {client.client_id} Test: Acc: {correct / samples:.4f}, Train: Acc: {correct2 / samples2:.4f}')
            metrics.extend([correct / samples, correct2/samples2, None])
            metrics[3 * i + 3] = loss * 1.0 / samples2
        metrics.append(time_cost)
        # 计算Test的准确率平均与方差并打印
        total_samples = sum(num_samples_list)
        total_correct = sum(total_correct_list)
        print(f'Average Test: Acc: {total_correct / total_samples:.4f}, Average Train: Acc: {sum(total_correct_list2) / sum(num_samples_list2):.4f}')
        print(f'Average Train loss: {sum(losses) / sum(num_samples_list2)}')
        print(f'Std Test Acc: {np.std([correct / samples for correct, samples in zip(total_correct_list, num_samples_list)]):.4f}')
        # 保存这些参数
        if total_correct / total_samples > self.best_acc:
            self.best_acc = total_correct / total_samples
            # 保存feature_extractor模型
            torch.save(self.global_model.state_dict(), f'./{self.save_dir}/{self.dataset_name}/Feature_Extractor.pth')
            # 保存全局Embedding模型
            torch.save(self.Embedding_Generator.state_dict(), f'./{self.save_dir}/{self.dataset_name}/Embedding_Generator.pth')
            # 保存全局HyperNetwork模型
            torch.save(self.HyperNetwork.state_dict(), f'./{self.save_dir}/{self.dataset_name}/HyperNetwork.pth')
            # 保存head模型
            torch.save(self.head.state_dict(), f'./{self.save_dir}/{self.dataset_name}/Head.pth')
            print(f'Best model saved. Acc: {self.best_acc:.4f}')
        # 保存客户端度量信息到 CSV 文件
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(metrics)