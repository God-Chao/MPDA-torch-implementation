from torch.utils.data import Dataset
import torch
import sys
import os

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 当前文件的父目录（需要调整路径的深度根据你的项目结构而定）
parent_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
# 将目标目录加入到 sys.path
sys.path.append(parent_dir)

class MovieLensDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]