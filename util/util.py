from model.model import FedCNN, FedUCNN


def return_model_args(args):
    # 返回转化后的模型
    str_model = args.model
    if str_model == "CNN" and args.retrain==False:
        if "MNIST" in args.dataset:
            args.model = FedCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "CIFAR10" in args.dataset:
            args.model = FedCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "SVHN" in args.dataset:
            args.model = FedCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "TINY" in args.dataset:
            args.model = FedCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
    elif str_model == "CNN" and args.retrain==True:
        if "MNIST" in args.dataset:
            args.model = FedUCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "CIFAR10" in args.dataset:
            args.model = FedUCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "SVHN" in args.dataset:
            args.model = FedUCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "TINY" in args.dataset:
            args.model = FedUCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
    else:
        raise NotImplementedError
    return args

