import torch
import torch.nn as nn
import torch.nn.functional as F

class FedCNN(nn.Module):
    def __init__(self, in_features, num_classes, dim, dim1=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim1),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(dim1, num_classes)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.conv2(out)
        # print(out.shape)
        out = torch.flatten(out, 1)
        # print(out.shape)
        out = self.fc1(out)
        # print(out.shape)
        out = self.fc(out)
        return out

class FedUCNN(nn.Module):
    def __init__(self, in_features, num_classes, dim, dim1=512):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim1),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(dim1, num_classes)

    def forward(self, x):
        out0 = self.conv1(x)
        out1 = self.conv2(out0)
        out2 = torch.flatten(out1, 1)
        out3 = self.fc1(out2)
        out = self.fc(out3)
        return out, [out0, out1, out2, out3]

class FeatureHeadSplit(nn.Module):
    def __init__(self, feature_extractor, head):
        super(FeatureHeadSplit, self).__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.head(out)
        return out
    
class HyperNetwork(nn.Module):
    def __init__(self, input_dim, num):
        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 该类的作用是将全局模型的embedding抽象出来，生成全局的Embendding
class Global_Embedding_Generator(nn.Module):
    def __init__(self, embedding_dim, num_classes, device):
        super(Global_Embedding_Generator, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.device = device
        # embedding层将num_classes个类别映射到num_classes个embedding_dim维度的向量
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim, device=self.device)

    def forward(self, x, label):
        # x是条件阀生成的全局图像，label是全局图像对应的标签
        # 生成input为0到num_classes-1的tensor，[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        input_tensor = torch.tensor(range(self.num_classes), device=self.device)
        # 生成全局的embedding, 10*embedding_dim
        embeddings = self.embedding(input_tensor)
        # 计算x与embedding的余弦相似度
        cosine = F.linear(F.normalize(x), F.normalize(embeddings))
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        softmax_value = F.log_softmax(cosine, dim=1)
        softmax_loss = one_hot * softmax_value
        softmax_loss = - torch.mean(torch.sum(softmax_loss, dim=1))
        return softmax_loss