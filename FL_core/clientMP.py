import copy
import torch
from FL_core.fedclient import Client
from torch.utils.data import DataLoader


class clientMP(Client):
    def __init__(self, args, cid):
        super().__init__(args, cid)
        self.embedding_dim = list(self.model.head.parameters())[0].shape[1] // 2
        self.HyperNetwork = copy.deepcopy(args.HyperNetwork)
        self.Global_Embedding_Generator = copy.deepcopy(args.Embedding_Generator)
        train_data = self.load_train_data()
        # 组织为batch
        self.train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_data = self.load_test_data()
        self.test_data = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        # 计算私人的数据分布
        self.class_ratio = self.compute_sample_per_class()
        self.retrain = args.retrain
        self.retrain = args.retrain
        self.lamda = args.lamda
        self.load = True
        if args.retrain == True and self.load == True:
            # 加载Embedding模型
            self.Global_Embedding_Generator.load_state_dict(torch.load("save_dir/{}_{}/{}/Embedding_Generator.pth".format(args.D_alpha,args.num_clients,args.dataset)))
            # 加载HyperNetwork模型
            self.HyperNetwork.load_state_dict(torch.load("save_dir/{}_{}/{}/HyperNetwork.pth".format(args.D_alpha,args.num_clients,args.dataset)))
        else:
            self.Global_Embedding_Generator_Opt = torch.optim.SGD(self.Global_Embedding_Generator.parameters(),
                                                               lr=self.learning_rate,
                                                               weight_decay=args.L2_weight)
            self.HyperNetwork_Opt = torch.optim.SGD(self.HyperNetwork.parameters(), 
                                                        lr=self.learning_rate, 
                                                        weight_decay=args.L2_weight)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.Personal_Input = torch.zeros(self.embedding_dim).to(self.device)
        self.Gloabl_Input = torch.zeros(self.embedding_dim).to(self.device)
        self.Global_Embedding_Generator_Frozen = copy.deepcopy(self.Global_Embedding_Generator)
    
    def train(self):
        # 加载训练数据
        train_data = self.train_data
        if train_data is None:
            raise Exception("Error: Failed to load training data.")
        for local_epoch in range(1, self.local_round + 1):
            # 训练模型
            self.model.train()
            for x, y in train_data:
                x = x.to(self.device)
                y = y.to(self.device) 
                influence_layer_wise_personal = self.HyperNetwork(self.Personal_Input)
                influence_layer_wise_global = self.HyperNetwork(self.Gloabl_Input)
                new_feature_ex_p = self.update_feature_ex_params(self.model.feature_extractor, influence_layer_wise_personal)
                new_feature_ex_g = self.update_feature_ex_params(self.model.feature_extractor, influence_layer_wise_global)
                # 获取通过feature_extractor得到的特征
                feature_p = new_feature_ex_p(x)
                feature_g = new_feature_ex_g(x)
                # 将特征拼接
                feature = torch.cat((feature_p, feature_g), 1)
                # 获取预测值
                output = self.model.head(feature)
                # 计算损失
                loss0 = self.loss(output, y)
                # 计算损失2
                loss1 = self.Global_Embedding_Generator(feature_g, y)
                emb = self.Global_Embedding_Generator_Frozen.embedding(y).detach()
                loss = loss0 + loss1 + torch.norm(feature_g - emb, p=2) * self.lamda
                # 计算梯度时不包括Global_Embedding_Generator_Frozen的参数
                self.opt.zero_grad()
                if self.retrain == False or self.load == False:
                    self.HyperNetwork.zero_grad()
                    self.Global_Embedding_Generator_Opt.zero_grad()
                loss.backward(retain_graph=True)
                self.opt.step()
                if self.retrain == False or self.load == False:
                    self.HyperNetwork_Opt.zero_grad()
                    self.Global_Embedding_Generator_Opt.zero_grad()
        print("Client: {}, Local Epoch: {}, Loss: {}".format(self.client_id, local_epoch, loss.item()))

    # 更新feature_ex的每层参数
    def update_feature_ex_params(self, feature_ex, influence_factors):
        updated_feature_ex = copy.deepcopy(feature_ex)
        param_index = 0
        for name, module in updated_feature_ex.named_modules():
            for p in module.parameters():
                if p.requires_grad:
                    layer_factor = influence_factors[param_index].item()
                    p.data.mul_(layer_factor)
            param_index += 1
        return updated_feature_ex
    
    # 更新base模型
    def update_from_global(self, global_model):
        for new_param, old_param in zip(global_model.parameters(), self.model.feature_extractor.parameters()):
            old_param.data = new_param.data.clone()
    
    # 更新head模型
    def update_head_from_global(self, global_head):
        for new_param, old_param in zip(global_head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()
    
    # 更新HyperNetwork模型
    def update_hypernetwork_from_global(self, global_hypernetwork):
        for new_param, old_param in zip(global_hypernetwork.parameters(), self.HyperNetwork.parameters()):
            old_param.data = new_param.data.clone()

    # 更新Embedding模型以及Condition输入
    def update_from_global_embedding(self, global_embedding):
        self.Gloabl_Input = torch.zeros(self.embedding_dim).to(self.device)
        self.Personal_Input = torch.zeros(self.embedding_dim).to(self.device)
        # 计算Embedding的输入更新
        update_embedding = self.Global_Embedding_Generator.embedding(
            torch.tensor(range(self.num_classes), device=self.device))
        # 根据客户端数据分布情况生成私人的embedding
        for (i, emb) in enumerate(update_embedding):
            self.Gloabl_Input += emb / self.num_classes
            self.Personal_Input += emb * self.class_ratio[i]
        for new_param, old_param in zip(global_embedding.parameters(), self.Global_Embedding_Generator.parameters()):
            old_param.data = new_param.data.clone()
        self.Global_Embedding_Generator_Frozen = copy.deepcopy(self.Global_Embedding_Generator)

    # 测试集测试
    def test(self, if_test = True):
        # 加载测试数据
        if if_test:
            test_data = self.test_data
        else:
            test_data = self.train_data
        if test_data is None:
            raise Exception("Error: Failed to load test data.")
        # 测试模型
        self.model.eval()
        correct = 0
        total = 0
        losses = 0
        with torch.no_grad():
            for x, y in test_data:
                x = x.to(self.device)
                y = y.to(self.device) 
                influence_layer_wise_personal = self.HyperNetwork(self.Personal_Input)
                influence_layer_wise_global = self.HyperNetwork(self.Gloabl_Input)
                new_feature_ex_p = self.update_feature_ex_params(self.model.feature_extractor, influence_layer_wise_personal)
                new_feature_ex_g = self.update_feature_ex_params(self.model.feature_extractor, influence_layer_wise_global)
                # 获取通过feature_extractor得到的特征
                feature_p = new_feature_ex_p(x)
                feature_g = new_feature_ex_g(x)
                # 将特征拼接
                feature = torch.cat((feature_p, feature_g), 1)
                # 获取预测值
                output = self.model.head(feature)
                total += y.size(0)
                correct += (torch.argmax(output, dim=1) == y).sum().item()
                if not if_test:
                    # 计算损失
                    loss0 = self.loss(output, y)
                    # 计算损失2
                    loss1 = self.Global_Embedding_Generator(feature_g, y)
                    emb = self.Global_Embedding_Generator_Frozen.embedding(y).detach()
                    loss = loss0 + loss1 + torch.norm(feature_g - emb, p=2) * self.lamda
                    losses += loss.item() * y.size(0)
        if if_test:
            return correct, total
        else:
            return correct, total, losses