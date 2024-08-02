class Config:
    def __init__(self):
        self.num_classes = 10  # 类别数
        self.num_clients = 10  # 客户端数
        self.server_round = 100  # 服务器迭代次数
        self.ratio = 1.0  # 选择客户端的比例
        self.batch_size = 1024  # 批大小
        self.local_round = 3  # 客户端迭代次数
        self.learning_rate = 0.005  # 学习率
        self.eval_every = 1  # 评估间隔
        self.train_ratio = 0.8  # 训练集比例
        self.D_alpha = 0.01  # dirichlet分布参数，越大越均衡
        self.partition = 'Dirichlet'  # 数据集划分方式，候选iid，exdir，Dirichlet, extreme-non-iid
        self.non_iid = True  # 是否非独立同分布
        self.balance = True  # 数据是否平衡
        self.ED_alpha = 100  # 拓展的dirichlet分布参数
        self.dataset = 'MNIST'  # 数据集，候选MNIST, CIFAR10, TINY, SVHN, CIFAR100, FMNIST
        if self.dataset == 'MNIST':
            self.num_classes = 10
            self.backbone = 'CNN'  # 主干网络
            self.channel = 1
        elif self.dataset == 'CIFAR10':
            self.num_classes = 10
            self.backbone = 'CNN'  # 主干网络
            self.server_round = 200
            self.channel = 3
        elif self.dataset == 'FMNIST':
            self.num_classes = 10
            self.backbone = 'CNN'  # 主干网络
            self.server_round = 200
            self.channel = 1
        elif self.dataset == 'SVHN':
            self.num_classes = 10
            self.backbone = 'CNN'  # 主干网络
            self.channel = 3
        self.L2_weight = 0.1  # L2正则化
        self.un_client = [5]  # 待遗忘客户端4,13,6,4
        self.remain_client = [4] # 保留客户端2,2,8,2
        self.head_aggregation = True  # Head是否聚合
        self.unlearning_round = 10000  # 遗忘知识蒸馏次数
        self.generator_noise_dim = 128  # 生成器噪声维度
        self.KL_alpa = 5 # KL散度系数,MNIST：5，CIFAR10：50，FMNIST：5
        self.KL_temperature = 2  # KL散度温度
        self.Att_alpa = 2 # Attention系数,1，0.5，6002

