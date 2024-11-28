import argparse
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from model import NCF
from sklearn.metrics import accuracy_score, roc_auc_score

def config_args():
    parser = argparse.ArgumentParser('train and test model for users in user_intersect.json')

    parser.add_argument('-processed_fp', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/processed', help='所有用户目录')
    parser.add_argument('-recall_num', type=int, default=50, help='云端召回用户数量')
    parser.add_argument('-recall_alg', type=str, default='random', help='云端召回算法')
    parser.add_argument('-log_fp', type=str, default='/home/chao/workspace/MPDA-implementation/log', help='日志目录')
    parser.add_argument('-data_fp', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/ratings.csv', help='训练集目录')
    parser.add_argument('-model', type=str, default='NCF', help='模型名称')
    parser.add_argument('-epochs', type=int, default=1, help='模型在每个用户训练集上微调的epoch')
    parser.add_argument('-device', type=str, default='cuda:5', help='训练模型的设备')
    parser.add_argument('-batch_size', type=int, default=64, help='batch大小')
    parser.add_argument('-embedding_dim', type=int, default=32, help='嵌入层纬度')
    parser.add_argument('-drop_out', type=float, default=0.1, help='drop out率')
    parser.add_argument('-lr', type=float, default=0.001, help='学习率')
    parser.add_argument('-model_save_path', type=str, default='/home/chao/workspace/MPDA-implementation/model/NCF/init_ncf.pth', help='模型保存路径')
    parser.add_argument('-timestamp', type=int, default=1225642324, help='划分的时间戳依据')
    parser.add_argument('-user_intersect_json', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/user_intersect.json', help='测试集用户json文件')


    args = parser.parse_args()
    return args

# 加载数据
class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user'].values, dtype=torch.long)
        self.items = torch.tensor(df['item'].values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def main():
    print(f'[{datetime.now()}] start transfer model NCF')
    args = config_args()
    print(f'{vars(args)}')

    # 设置超参数
    batch_size = args.batch_size
    epochs = args.epochs
    embedding_dim = args.embedding_dim
    dropout = args.drop_out
    num_users, num_items = 138493, 27278
    device = torch.device(args.device)
    learning_rate = args.lr

    model = NCF(num_users, num_items, embedding_dim, dropout).to(device)

    with open(args.user_intersect_json, 'r', encoding='utf-8') as f:
        test_users = json.load(f)
    print(f'user-intersect.json has been loaded')

    '''
    总共有四种模式
    Cloud: 在所用用户训练集上训练一个epoch
    Local: 使用Cloud模型作为初始模型在本地训练集训练一个epoch
    Local+: 使用Cloud模型作为初始模型在本地训练集+增强数据上训练一个epoch
    MPDA-: 在增强数据集上训练一个epoch
    MPDA: 在本地数据+增强数据训练一个epoch
    '''
    for index, user in enumerate(test_users):
        print(f'[{datetime.now()}] iter {index}/{len(test_users)}, checking user[{user}]')

        # 加载初始模型
        model.load_state_dict(torch.load(args.model_save_path))
        print(f'init model has been loaded')

        # 加载本地数据集
        local_data = pd.read_csv()



if __name__ == '__main__':
    main()
    