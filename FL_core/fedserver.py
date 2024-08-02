import copy

import numpy as np


class Server(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.num_clients = args.num_clients
        self.server_round = args.server_round
        self.global_model = copy.deepcopy(args.model)
        self.ratio = args.ratio
        self.clients = []
        self.selected_clients = []
        self.eval_every = args.eval_every
        self.best_acc = 0.0
        self.dataset_name = args.dataset
        self.un_client = []

    def select_clients(self):
        self.selected_clients = np.random.choice(self.clients, self.num_clients-len(self.un_client), replace=False)
        # 排序
        self.selected_clients = sorted(self.selected_clients, key=lambda client: client.client_id)
