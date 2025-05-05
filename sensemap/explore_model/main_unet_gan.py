import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MapReconstructionDataset
from UNet_transformer_new import UNet as UNet_transformer
from UNet import UNet
from SenseMapNet import SenseMapNet
from critic_model import CriticModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2
from train_gan_new import train_model
import os

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters: {}, Trainable: {}'.format(total_num, trainable_num))


def start_train_model(gen, save_dir):
    torch.cuda.empty_cache()
    get_parameter_number(gen)

    # 确定使用设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # critic = FFCResNet(Bottleneck, [2, 2, 2, 2]).to(device)
    critic = CriticModel()

    # 如果有多个GPU，使用DataParallel进行多GPU训练
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        gen = nn.DataParallel(gen)  # 使用DataParallel来包装模型
        critic = nn.DataParallel(critic)

    gen.to(device)  # 将模型移动到GPU/CPU
    critic.to(device)

    # 训练模型
    train_model(gen, critic, dataloaders, num_epochs=200, device=device, save_dir=save_dir)

if __name__ == "__main__":
    # 设置路径
    dataset_dir = 'dataset3'

    # 创建数据集和数据加载器
    train_dataset = MapReconstructionDataset(dataset_dir + "/train/")
    test_dataset = MapReconstructionDataset(dataset_dir + "/test/")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)  

    dataloaders = {'train': train_loader, 'test': test_loader}

    # # 初始化模型
    # model = UNet()
    # start_train_model(model, save_dir = "explore_model/model_logs/unet/")

    model = SenseMapNet(base=1)
    start_train_model(model, save_dir = "explore_model/model_logs/ffc/")

    # model = UNet_transformer(base=1)
    # start_train_model(model, save_dir = "explore_model/model_logs/transformer/")

    # model = FFCResNet(Bottleneck, [3, 4, 6, 3])
    # start_train_model(model, save_dir = "explore_model/model_logs/ffc/")
