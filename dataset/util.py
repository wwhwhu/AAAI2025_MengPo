import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import ujson
from config.config import Config

config = Config()
train_ratio = config.train_ratio


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None,
                  D_alpha=config.D_alpha, ED_alpha=config.ED_alpha, batch_size=config.batch_size):
    global idx_batch
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    # 保证每个客户端至少有一个批次的数据用于测试
    least_samples = int(min(batch_size / (1 - train_ratio), len(dataset_label) / num_clients / 2))
    dataidx_map = {}

    if not niid:
        partition = 'iid'
        class_per_client = num_classes

    if partition == 'iid':
        idxs = np.arange(len(dataset_label))
        # 获取每个类别的索引
        idx_for_each_class = [idxs[dataset_label == i] for i in range(num_classes)]
        # 每个客户端分配的类别数
        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            # 对于第i类数据选取所有的客户端
            selected_clients = [client for client in range(num_clients) if class_num_per_client[client] > 0]
            if not selected_clients:
                break
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]
            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            # 每个客户端平均应该分配的第i类的数据量
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                # 数据量不balance时，每个客户端分配的数据量在num_per的0.1倍和num_per之间
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map:
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    # 按照Dirichlet分布分配数据
    elif partition == "Dirichlet":
        print(least_samples)
        K = num_classes
        N = len(dataset_label)
        min_size = 0
        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                # 空列表保存每个类别的索引
                idx_k = np.where(dataset_label == k)[0]
                # 打乱索引
                np.random.shuffle(idx_k)
                # 使用 Dirichlet 分布生成 num_clients 个值，这些值作为分配比例
                proportions = np.random.dirichlet(np.repeat(D_alpha, num_clients))
                # 调整生成的比例，确保每个客户端的数据量不超过N / num_clients
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                # 使用累积和计算每个客户端应分配的样本数，得到样本分割点。
                print(proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # 分配样本
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

    # This strategy comes from https://arxiv.org/abs/2311.03154
    elif partition == 'exdir':
        C = class_per_client
        min_size_per_label = 0
        # 每个类别至少需要的客户端数量，确保每个类别至少有一部分数据分配到多个客户端中
        min_require_size_per_label = max(C * num_clients // num_classes // 2, 1)
        clientidx_map = {}

        # 通过循环分配，每个客户端被分配到 C 个类别，确保每个类别至少分配到 min_require_size_per_label 个客户端。
        while min_size_per_label < min_require_size_per_label:
            for k in range(num_classes):
                clientidx_map[k] = []
            for i in range(num_clients):
                labelidx = np.random.choice(range(num_classes), C, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])

        dataidx_map = {}
        y_train = dataset_label
        min_size = 0
        min_require_size = 10
        K = num_classes
        N = len(y_train)

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(ED_alpha, num_clients))
                proportions = np.array(
                    [p * (len(idx_j) < N / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in
                     enumerate(zip(proportions, idx_batch))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients - 1):
                        proportions[w] = len(idx_k)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]
    elif partition == 'extreme-non-iid':
        # 每个Client对应一个类别
        if num_clients != num_classes:
            raise ValueError("For extreme non-iid, the number of clients should be equal to the number of classes.")
        else:
            for i in range(num_clients):
                dataidx_map[i] = np.where(dataset_label == i)[0]    
    else:
        raise NotImplementedError(f"Partition method '{partition}' is not implemented.")

    # 获得每个客户端的数据，包括数据和标签
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):
    # 初始化训练数据和测试数据的列表
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}
    for i in range(len(y)):
        # 对每个客户端的数据进行分割
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)
        # 将分割后的数据添加到相应的列表中
        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))
    # 输出数据集的总体情况
    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    # 删除原始数据以释放内存
    del X, y
    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, statistic):
    print("Saving dataset.\n")
    configC = Config()
    configF = {
        'num_clients': configC.num_clients,
        'num_classes': configC.num_classes,
        'non_iid': configC.non_iid,
        'balance': configC.balance,
        'partition': configC.partition,
        'Size of samples for labels in clients': statistic,
        'D_alpha': configC.D_alpha,
        'ED_alpha': configC.ED_alpha,
        'batch_size': configC.batch_size,
    }
    # 不存在则创建目录
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    # 保存训练数据
    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    # 保存测试数据
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    # 保存配置文件
    with open(config_path, 'w') as f:
        ujson.dump(configF, f)
    print("Finish generating dataset.\n")


# 可视化数据集
def visualize_client_data_distribution_1(train_data, num_clients, num_classes, output_file):
    font = {
        'family': 'Arial',  # 字体样式
        'weight': 'normal',   # 字体粗细
        'size': 20          # 字体大小
    }
    plt.style.use('ggplot')
    plt.rc('font', **font)  # 应用字体设置
    font = {'family':'Arial', 'weight': 'normal', 'size': 35, 'color': 'black'}
    # 计算每个客户端在每个类别上的样本数量
    distribution = np.zeros((num_clients, num_classes))
    for client_id, data in enumerate(train_data):
        for label in data['y']:
            distribution[client_id, label] += 1
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(8, 5))
    for client_id in range(distribution.shape[0]):
        for class_id in range(distribution.shape[1]):
            plt.scatter(client_id, class_id, s=distribution[client_id, class_id] * 1.5, c='red', alpha=0.6,                        edgecolors='w', linewidth=0.5)
    # 设置x轴和Y轴字体颜色
    plt.tick_params(axis='x', colors='black')
    plt.tick_params(axis='y', colors='black')
    # 添加虚拟的散点以显示在图例中
    plt.scatter([], [], s=100, c='red', alpha=0.6, edgecolors='w', linewidth=0.2, label='Sample Size')
    legend = plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc=1, prop = {'size':20})
    legend.get_frame().set_facecolor('white')  # 设置背景色
    legend.get_frame().set_alpha(0.8)              # 设置透明度
    plt.ylabel('Class', font)
    plt.xlabel('Client', font)
    # plt.title(f'$\zeta$ ={config.D_alpha} Data Distribution', fontsize=25, fontweight='normal')
    plt.yticks(range(num_classes), [f'{i}' for i in range(num_classes)])
    plt.xticks(range(num_clients), [f'{i}' for i in range(num_clients)])
    plt.grid(True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()

def visualize_client_data_distribution(num_clients, num_classes, output_file):
    font = {'family':'Arial', 'weight': 'normal', 'size': 35, 'color': 'black'}
    # 获取train_data，读取文件夹下的所有文件
    train_data = []
    for i in range(num_clients):
        with open(f'data_dir/{config.D_alpha}_{num_clients}/{config.dataset}/train/{i}.npz', 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].item()
            train_data.append(data)
    # 计算每个客户端在每个类别上的样本数量
    distribution = np.zeros((num_clients, num_classes))
    for client_id, data in enumerate(train_data):
        for label in data['y']:
            distribution[client_id, label] += 1
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(8, 5))
    for client_id in range(distribution.shape[0]):
        for class_id in range(distribution.shape[1]):
            plt.scatter(client_id, class_id, s=distribution[client_id, class_id] * 1.5, c='red', alpha=0.6,                        edgecolors='w', linewidth=0.5)
    # 设置x轴和Y轴字体颜色
    plt.tick_params(axis='x', colors='black')
    plt.tick_params(axis='y', colors='black')
    # 添加虚拟的散点以显示在图例中
    plt.scatter([], [], s=100, c='red', alpha=0.6, edgecolors='w', linewidth=0.2, label='Sample Size')
    legend = plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc=1, prop = {'size':20})
    legend.get_frame().set_facecolor('white')  # 设置背景色
    legend.get_frame().set_alpha(0.8)              # 设置透明度
    plt.ylabel('Class', font)
    plt.xlabel('Client', font)
    # plt.title(f'$\zeta$ ={config.D_alpha} Data Distribution', fontsize=25, fontweight='normal')
    plt.yticks(range(num_classes), [f'{i}' for i in range(num_classes)])
    plt.xticks(range(num_clients), [f'{i}' for i in range(num_clients)])
    plt.grid(True)
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    font = {
        'family': 'Arial',  # 字体样式
        'weight': 'normal',   # 字体粗细
        'size': 20          # 字体大小
    }
    plt.style.use('ggplot')
    plt.rc('font', **font)  # 应用字体设置
    # 可视化数据集
    visualize_client_data_distribution(config.num_clients, config.num_classes, f"data_dir/{config.D_alpha}_10/{config.dataset}/train_distribution.pdf")