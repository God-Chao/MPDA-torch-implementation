import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from model import NCF
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import sys
import numpy as np
import copy

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 当前文件的父目录（需要调整路径的深度根据你的项目结构而定）
parent_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
# 将目标目录加入到 sys.path
sys.path.append(parent_dir)
from utils.utils import utils
from Dataset import MovieLensDataset


class train_and_test_utils():
    # 模型model在训练集train_data上训练epochs次
    def train_model(model, train_data, criterion, optimizer, batch_size, device, epochs):
        # 数据预处理
        train_data['label'] = (train_data['rating'] >= 4).astype(int)
        # 去除其他列
        train_data = train_data[['user_id', 'item_id', 'label']]
        train_dataset = MovieLensDataset(train_data)
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        start_time = datetime.now()
        print(f'[{start_time}] start train model')
        # 训练
        for epoch in range(epochs):
            model.train()
            for index, (batch_user, batch_item, batch_label) in enumerate(train_loader):
                print(f'[{datetime.now()}] train {index}/{len(train_loader)}')
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
        
        end_time = datetime.now()
        print(f'[{end_time}] train fininshed, time cost {(end_time-start_time).total_seconds()} s')
        return None
    
    def test_model(model, test_data, batch_size, device):
        # 数据预处理
        test_data['label'] = (test_data['rating'] >= 4).astype(int)
        # 去除其他列
        test_data = test_data[['user_id', 'item_id', 'label']]
        test_dataset = MovieLensDataset(test_data)
        # 创建数据加载器
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        start_time = datetime.now()
        print(f'[{start_time}] start test model')
        model.eval()
        all_labels = []
        all_preds = []
        for index, (batch_user, batch_item, batch_label) in enumerate(test_loader):
            print(f'[{datetime.now()}] test {index}/{len(test_loader)}')
            batch_user = batch_user.to(device)
            batch_item = batch_item.to(device)
            batch_label = batch_label.to(device).unsqueeze(1)
            
            preds = model(batch_user, batch_item)
            all_labels.extend(batch_label.cpu().detach().numpy())
            all_preds.extend(preds.cpu().detach().numpy())

        end_time = datetime.now()
        print(f'[{end_time}] test fininshed, time cost {(end_time-start_time).total_seconds()} s')
        # 计算准确率
        if len(np.unique(all_labels)) < 2:
            print("Warning: Only one class present in labels. AUC is not defined.")
            return 0.5
        all_preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
        all_labels_flat = [int(l) for l in all_labels]
        accuracy = accuracy_score(all_labels_flat, all_preds_binary)
        
        # 计算 AUC
        auc = roc_auc_score(all_labels_flat, all_preds)
        print(f"Test Accuracy: {accuracy:.4f}, Test AUC: {auc:.4f}")
        return auc
    
    def train_model_with_dataset(model, criterion, optimizer, batch_size, train_data, device):
        train_data['label'] = (train_data['rating'] >= 4).astype(int)
        train_data = train_data[['user_id', 'item_id', 'label']]
        train_dataset = MovieLensDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model.train()
        for batch_user, batch_item, batch_label in train_loader:
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
        return None
    
    def test_model_with_dataset(model, test_data, batch_size, device):
        test_data['label'] = (test_data['rating'] >= 4).astype(int)
        test_data = test_data[['user_id', 'item_id', 'label']]
        test_dataset = MovieLensDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        all_labels = []
        all_preds = []
        for batch_user, batch_item, batch_label in test_loader:
            batch_user = batch_user.to(device)
            batch_item = batch_item.to(device)
            batch_label = batch_label.to(device).unsqueeze(1)
            
            preds = model(batch_user, batch_item)
            all_labels.extend(batch_label.cpu().detach().numpy())
            all_preds.extend(preds.cpu().detach().numpy())

        if len(np.unique(all_labels)) < 2:
            print("Warning: Only one class present in labels. AUC is not defined.")
            return 0.5
        all_labels_flat = [int(l) for l in all_labels]
        auc = roc_auc_score(all_labels_flat, all_preds)
        return auc
    
    def incremental_training(model, criterion, test_data, augumented_users, device, batch_size, best_auc):
        current_model = copy.deepcopy(model)  # 当前模型
        print(f"Initial AUC: {best_auc}")
        num_selected_users = 0

        for index, augumented_user in enumerate(augumented_users):
            print(f"Training on augumented user {augumented_user} {index}/{len(augumented_users)} current_auc = {best_auc}")
            augumented_user_trainset = utils.get_user_trainset(augumented_user)
            temp_model = copy.deepcopy(current_model)  # 临时模型
            temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.001)

            # 在增强用户的数据上训练一个 epoch
            train_and_test_utils.train_model_with_dataset(temp_model, criterion, temp_optimizer, batch_size, augumented_user_trainset, device)

            # 评估增强后的模型
            temp_auc = train_and_test_utils.test_model_with_dataset(temp_model, test_data, batch_size, device)
            print(f"User {augumented_user} AUC after training: {temp_auc}")

            if temp_auc > best_auc:
                # 如果 AUC 提升，保留增强的模型
                print(f"User {augumented_user} improves AUC. Keeping the model.")
                num_selected_users += 1
                current_model = temp_model
                best_auc = temp_auc
            else:
                # 如果 AUC 没有提升，回退模型
                print(f"User {augumented_user} does not improve AUC. Reverting changes.")

        return current_model, best_auc, num_selected_users
        