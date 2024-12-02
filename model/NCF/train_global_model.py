import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from model import NCF
from Dataset import MovieLensDataset
from train_and_test_utils import train_and_test_utils
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import sys

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 当前文件的父目录（需要调整路径的深度根据你的项目结构而定）
parent_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
# 将目标目录加入到 sys.path
sys.path.append(parent_dir)
from utils.utils import utils

def config_args():
    parser = argparse.ArgumentParser('train global model')

    parser.add_argument('-epochs', type=int, default=1, help='模型在每个训练集上训练的epoch')
    parser.add_argument('-device', type=str, default='cuda:5', help='训练模型的设备')
    parser.add_argument('-batch_size', type=int, default=64, help='batch大小')
    parser.add_argument('-lr', type=float, default=0.001, help='学习率')
    parser.add_argument('-model_save_path', type=str, default=os.path.join(utils.get_model_path_by_name('NCF'),'init_model.pth'), help='模型保存路径')

    args = parser.parse_args()
    return args

# 加载数据


def main():
    print(f'[{datetime.now()}] start train global model NCF')
    args = config_args()
    print(f'{vars(args)}')

    # 设置超参数
    batch_size = args.batch_size
    epochs = args.epochs
    device = torch.device(args.device)
    learning_rate = args.lr

    # 加载训练集和测试集
    train_data, test_data = utils.get_train_test_data()
    print(f'train and test data has been loaded')

    # 加载模型
    model = NCF(device).to(device)
    print(f'model has been loaded')

    # 定义损失函数和优化器
    criterion = nn.BCELoss() # 二元交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f'criterion and optimizer had been loaded')

    # 训练
    train_and_test_utils.train_model(model, train_data, criterion, optimizer, batch_size, device, epochs)

    # 训练
    train_and_test_utils.test_model(model, test_data, batch_size, device)
    
    # 保存和加载模型
    torch.save(model.state_dict(), args.model_save_path)
    print(f'model has been saved to {args.model_save_path}')

if __name__ == '__main__':
    main()