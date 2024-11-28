import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from model import NCF
from sklearn.metrics import accuracy_score, roc_auc_score

def config_args():
    parser = argparse.ArgumentParser('train global model')

    parser.add_argument('-processed_fp', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/processed', help='所有用户目录')
    parser.add_argument('-data_fp', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/ratings.csv', help='训练集目录')
    parser.add_argument('-model', type=str, default='NCF', help='模型名称')
    parser.add_argument('-epochs', type=int, default=1, help='模型在每个训练集上训练的epoch')
    parser.add_argument('-device', type=str, default='cuda:5', help='训练模型的设备')
    parser.add_argument('-batch_size', type=int, default=64, help='batch大小')
    parser.add_argument('-embedding_dim', type=int, default=32, help='嵌入层纬度')
    parser.add_argument('-drop_out', type=float, default=0.1, help='drop out率')
    parser.add_argument('-lr', type=float, default=0.001, help='学习率')
    parser.add_argument('-model_save_path', type=str, default='/home/chao/workspace/MPDA-implementation/model/NCF/init_ncf.pth', help='模型保存路径')
    parser.add_argument('-timestamp', type=int, default=1225642324, help='划分的时间戳依据')

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
    print(f'[{datetime.now()}] start train global model NCF')
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

    # 数据预处理
    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(args.data_fp, header=0, names=column_names)
    print(f'all data has been loaded')

    # 编码用户和物品id
    user_encoder = LabelEncoder()
    data['user'] = user_encoder.fit_transform(data['user_id'])

    item_encoder = LabelEncoder()
    data['item'] = item_encoder.fit_transform(data['item_id'])

    # 二值化评分：>=4为1，<4为0
    data['label'] = (data['rating'] >= 4).astype(int)

    num_users = data['user'].nunique()
    num_items = data['item'].nunique()

    # 分割训练集和测试集
    train_data = data[data['timestamp'] < args.timestamp]
    test_data = data[data['timestamp'] >= args.timestamp]
    print(f'train and test data has been splitted, len(train_data)={len(train_data)}, len(test_data)={len(test_data)}')

    # 去除其他列
    train_data = train_data[['user', 'item', 'label']]
    test_data = test_data[['user', 'item', 'label']]

    train_dataset = MovieLensDataset(train_data)
    test_dataset = MovieLensDataset(test_data)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = NCF(num_users, num_items, embedding_dim, dropout).to(device)
    print(f'model has been loaded')

    # 定义损失函数和优化器
    criterion = nn.BCELoss() # 二元交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f'start train')
    # 训练
    for epoch in range(epochs):
        model.train()
        for index, (batch_user, batch_item, batch_label) in enumerate(train_loader):
            print(f'[{datetime.now()}] {index}/{len(train_loader)}')
            batch_user = batch_user.to(device)
            batch_item = batch_item.to(device)
            batch_label = batch_label.to(device).unsqueeze(1)  # 形状: (batch_size, 1)
            
            # 前向传播
            preds = model(batch_user, batch_item)
            
            # 计算损失
            loss = criterion(preds, batch_label)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    print(f'[{datetime.now()}] train fininshed')

    print(f'start test global model')
    model.eval()
    all_labels = []
    all_preds = []
    avg_loss = 0.
    for index, (batch_user, batch_item, batch_label) in enumerate(test_loader):
        batch_user = batch_user.to(device)
        batch_item = batch_item.to(device)
        batch_label = batch_label.to(device).unsqueeze(1)
        
        preds = model(batch_user, batch_item)
        all_labels.extend(batch_label.cpu().detach().numpy())
        all_preds.extend(preds.cpu().detach().numpy())
        avg_loss += criterion(preds, batch_label)

    # 计算准确率
    all_preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
    all_labels_flat = [int(l) for l in all_labels]
    accuracy = accuracy_score(all_labels_flat, all_preds_binary)
    avg_loss /= len(test_loader)
    
    # 计算 AUC
    auc = roc_auc_score(all_labels_flat, all_preds)
    print(f"Test average Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, Test AUC: {auc:.4f}")

    # 保存和加载模型
    torch.save(model.state_dict(), args.model_save_path)
    print(f'model has been saved!')

if __name__ == '__main__':
    main()
    