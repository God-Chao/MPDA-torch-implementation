import argparse
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
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
from utils.match import match
from model import NCF
from train_and_test_utils import train_and_test_utils
from Dataset import MovieLensDataset


def config_args():
    parser = argparse.ArgumentParser('train and test model for users in user_intersect.json')

    parser.add_argument('-recall_num', type=int, default=100, help='云端召回用户数量')
    parser.add_argument('-recall_alg', type=str, default='random', help='云端召回算法')
    parser.add_argument('-epochs', type=int, default=1, help='模型在每个用户训练集上微调的epoch')
    parser.add_argument('-device', type=str, default='cuda:5', help='训练模型的设备')
    parser.add_argument('-batch_size', type=int, default=64, help='batch大小')
    parser.add_argument('-task_index', type=int, default=0, help='任务并行工作下标')
    parser.add_argument('-num_task', type=int, default=utils.get_num_task(), help='总并行个数')
    parser.add_argument('-lr', type=float, default=0.001, help='学习率')


    args = parser.parse_args()
    return args

'''
总共有四种模式
Cloud: 在所用用户训练集上训练一个epoch
Local: 使用Cloud模型作为初始模型在本地训练集训练一个epoch
Local+: 使用Cloud模型作为初始模型在本地训练集+增强数据上训练一个epoch
MPDA-: 使用Cloud模型作为初始模型在增强数据集上增量训练
MPDA: 使用Cloud模型作为初始模型在本地数据+增强数据进行增量训练
DIPS(Device-Item-Pair-Similarity): 在本地模型上
'''
def main():
    print(f'[{datetime.now()}] start transfer model NCF')
    args = config_args()
    print(f'{vars(args)}')

    # 设置超参数
    recall_num = args.recall_num
    recall_alg = args.recall_alg
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = torch.device(args.device)
    batch_size = args.batch_size
    task_index = args.task_index
    num_task = args.num_task

    # 加载测试用户和划分task_index
    test_users = utils.get_user_intersect()
    test_users = np.array_split(test_users, num_task)[task_index]
    print(f'test users have been loaded, len = {len(test_users)}')
    print(f'test users = {test_users}')
    
    user_id_list =[]
    num_selected_users_list =[]
    num_train_samples_list = []
    num_test_samples_list =[]
    cloud_list = []
    local_list = []
    local_plus_list = []
    mpda_minus_list = []
    mpda_list = []

    start_time = datetime.now()
    print(f'[{start_time}] start test on test users')
    for index, user in enumerate(test_users):
        print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)}')
        
        # 加载初始模型
        init_model = NCF(device).to(device)
        init_model.load_state_dict(torch.load(utils.get_init_model_path('NCF')))

        criterion = nn.BCELoss() # 二元交叉熵损失

        user_id_list.append(user)
        augumented_users = []
        if recall_alg == 'random':
            augumented_users = match.random_match(recall_num)
            print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)} recall augumented users by random, augumented_users = {augumented_users}')
        elif recall_alg == 'interaction':
            augumented_users = match.match_by_interaction(recall_num)
            print(f'[{datetime.now()}] start test user {user} {index}/{len(test_users)} recall augumented users by interaction, augumented_users = {augumented_users}')

        test_data = utils.get_user_testset(user)
        train_data = utils.get_user_trainset(user)

        num_train_samples_list.append(len(train_data))
        num_test_samples_list.append(len(test_data))

        # Cloud: 直接用初始模型在本地测试集上测试
        print(f'[{datetime.now()}] user = {user} Cloud {index}/{len(test_users)}')
        cloud_auc = train_and_test_utils.test_model_with_dataset(init_model, test_data, batch_size, device)
        cloud_list.append(cloud_auc)

        # Local: 使用Cloud模型作为初始模型在本地训练集训练一个epoch
        optimizer = torch.optim.Adam(init_model.parameters(), lr=lr)
        print(f'[{datetime.now()}] user = {user} Local {index}/{len(test_users)}')
        train_and_test_utils.train_model_with_dataset(init_model, criterion, optimizer, batch_size, train_data, device)
        local_auc = train_and_test_utils.test_model_with_dataset(init_model, test_data, batch_size, device)
        local_list.append(local_auc)

        # Local+: 使用Cloud模型作为初始模型在本地训练集+增强数据上训练一个epoch
        print(f'[{datetime.now()}] user = {user} Local+ {index}/{len(test_users)}')
        init_model.load_state_dict(torch.load(utils.get_init_model_path('NCF')))
        optimizer = torch.optim.Adam(init_model.parameters(), lr=lr)
        for augumented_user in augumented_users:
            augumented_trainset = utils.get_user_trainset(augumented_user)
            train_and_test_utils.train_model_with_dataset(init_model, criterion, optimizer, batch_size, augumented_trainset, device)
        local_plus_auc = train_and_test_utils.test_model_with_dataset(init_model, test_data, batch_size, device)
        local_plus_list.append(local_plus_auc)

        # MPDA-: 使用Cloud模型作为初始模型在增强数据集上增量训练
        # MPDA: 使用Cloud模型作为初始模型在本地数据+增强数据进行增量训练
        print(f'[{datetime.now()}] user = {user} MPDA- {index}/{len(test_users)}')
        init_model.load_state_dict(torch.load(utils.get_init_model_path('NCF')))
        optimizer = torch.optim.Adam(init_model.parameters(), lr=lr)

        best_auc = cloud_auc # 记录最好的auc
        current_model = NCF(device).to(device)
        current_model.load_state_dict(torch.load(utils.get_init_model_path('NCF')))

        # 在增强用户上做增强训练，只选择提升了模型性能的增强用户
        current_model, best_auc, num_selected_users = train_and_test_utils.incremental_training(init_model, criterion, test_data, augumented_users, device, batch_size, cloud_auc)
        optimizer = torch.optim.Adam(current_model.parameters(), lr=lr)
        mpda_minus_list.append(best_auc)
        # 在本地训练集上训练一个epoch
        train_and_test_utils.train_model_with_dataset(current_model, criterion, optimizer, batch_size, train_data, device)
        mpda_auc = train_and_test_utils.test_model_with_dataset(current_model, test_data, batch_size, device)
        mpda_list.append(mpda_auc)
        num_selected_users_list.append(num_selected_users)

    log_fp = os.path.join(utils.get_log_path(), 'transfer_movielens_ncf_50_random')
    utils.result_to_xlsx(user_id_list, num_selected_users_list, num_train_samples_list, num_test_samples_list, cloud_list, local_list, local_plus_list, mpda_minus_list, mpda_list, log_fp, task_index)


if __name__ == '__main__':
    main()
    