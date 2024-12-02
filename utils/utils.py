import os
import pandas as pd
import yaml
from openpyxl import Workbook
import json
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np

class utils():
    def get_config_by_name(name):
        # 获取当前 Python 文件的绝对路径
        script_path = os.path.abspath(__file__)
        # 获取当前 Python 文件所在的目录
        script_directory = os.path.dirname(script_path)
        # 获取当前 Python 文件所在目录的上一级目录
        parent_directory = os.path.dirname(script_directory)
        with open(os.path.join(parent_directory, 'config.yml'), "r") as file:
            config = yaml.safe_load(file)
        # 访问变量
        variable = config[name]
        return variable

    def get_home_path():
        return utils.get_config_by_name('home_path')
    
    def get_timestamp():
        return utils.get_config_by_name('timestamp')
    
    def get_random_seed():
        return utils.get_config_by_name('random_seed')

    def get_num_task():
        return utils.get_config_by_name('num_task')
    
    def get_num_recall_item_pairs():
        return utils.get_config_by_name('num_recall_item_pairs')
    
    def get_all_users():
        file_path = os.path.join(utils.get_movielens_data_path(), 'all_users_id.json')
        with open(file_path, "r", encoding="utf-8") as file:
            users = json.load(file)
        return users
    
    def get_user_with_train():
        file_path = os.path.join(utils.get_movielens_data_path(), 'user_with_train.json')
        with open(file_path, "r", encoding="utf-8") as file:
            users = json.load(file)
        return users

    def get_train_test_data():
        data = utils.get_raw_ratings()
        timestamp = utils.get_timestamp()
        trainset = data[data['timestamp'] < timestamp]
        testset = data[data['timestamp'] >= timestamp]
        return trainset, testset
    
    def get_raw_ratings():
        data_path = os.path.join(utils.get_movielens_data_path(), 'ratings.csv')
        column_names = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_csv(data_path, header=0, names=column_names)
        return data

    def get_movielens_data_path():
        home_path = utils.get_home_path()
        return os.path.join(home_path, 'data/MovieLens/ml-20m')
    
    def get_log_path():
        home_path = utils.get_home_path()
        return os.path.join(home_path, 'log')
    
    def get_model_path():
        home_path = utils.get_home_path()
        return os.path.join(home_path, 'model')
    
    def get_model_path_by_name(model_name):
        model_path = utils.get_model_path()
        return os.path.join(model_path, model_name)
    
    def get_saved_model_path_by_name(model_name):
        model_path = utils.get_model_path_by_name(model_name)
        return os.path.join(model_path, 'init_model.pth')
    
    def get_init_model_path(model_name):
        model_path = utils.get_model_path_by_name(model_name)
        return os.path.join(model_path, 'init_model.pth')

    def get_processed_path():
        home_path = utils.get_movielens_data_path()
        return os.path.join(home_path, 'processed')
    
    def get_user_path(user):
        return os.path.join(utils.get_processed_path(), user)

    def get_user_raw_data(user):
        user = str(user)
        user_path = utils.get_user_path(user)
        raw_data_fp = os.path.join(user_path, 'user_'+str(user)+'_raw_data.csv')
        column_names = ['user_id', 'item_id', 'rating', 'timestamp']
        data = pd.read_csv(raw_data_fp, header=0, names=column_names)
        return data
    
    def get_user_seq_data(user):
        user_path = utils.get_user_path(user)
        seq_data_fp = os.path.join(user_path, 'user_'+str(user)+'_seq.csv')
        return pd.read_csv(seq_data_fp)
    
    def get_user_trainset(user):
        raw_data = utils.get_user_raw_data(user)
        return raw_data[raw_data['timestamp'] < utils.get_timestamp()]

    def get_user_testset(user):
        raw_data = utils.get_user_raw_data(user)
        return raw_data[raw_data['timestamp'] >= utils.get_timestamp()]
    
    def get_user_intersect():
        data_path = utils.get_movielens_data_path()
        file_path = os.path.join(data_path, 'user_intersect.json')
        with open(file_path, "r", encoding="utf-8") as file:
            users = json.load(file)
        return users

    def get_user_mapping_path():
        user_path = os.path.join(utils.get_movielens_data_path(), 'user_mapping.json')
        return user_path
    
    def get_item_mapping_path():
        item_path = os.path.join(utils.get_movielens_data_path(), 'item_mapping.json')
        return item_path
    
    def get_user_mapping_file():
        user_mapping_path = utils.get_user_mapping_path()
        with open(user_mapping_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    def get_item_mapping_file():
        item_mapping_path = utils.get_item_mapping_path()
        with open(item_mapping_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    def get_user_mapping_id(user_mapping_file, user_id):
        if isinstance(user_id, int):  # 如果是单个值
            return user_mapping_file[str(user_id)]
        elif isinstance(user_id, (list, np.ndarray)):  # 如果是列表或 NumPy 数组
            return [user_mapping_file[str(uid)] for uid in user_id]
        elif isinstance(user_id, torch.Tensor):  # 如果是 Tensor
            user_id = user_id.tolist() 
            return torch.tensor([user_mapping_file[str(uid)] for uid in user_id])
    
    def get_item_mapping_id(item_mapping_file, item_id):
        if isinstance(item_id, int):  # 如果是单个值
            return item_mapping_file[str(item_id)]
        elif isinstance(item_id, (list, np.ndarray)):  # 如果是列表或 NumPy 数组
            return [item_mapping_file[str(iid)] for iid in item_id]
        elif isinstance(item_id, torch.Tensor):  # 如果是 Tensor
            item_id = item_id.tolist() 
            return torch.tensor([item_mapping_file[str(iid)] for iid in item_id])
        
    def load_init_model(model, model_name):
        init_model_path = utils.get_init_model_path(model_name)
        model.load_state_dict(torch.load(init_model_path))
        return model
    
    def result_to_xlsx(user_id_list, num_selected_users_list, num_train_samples_list, 
    num_test_samples_list, cloud_list, local_list, local_plus_list, mpda_minus_list, mpda_list, log_fp, task_index):
        names = ['user_id', 'num_selected_users', 'num_train_samples', 'num_test_samples', 'Cloud', 'Local', 'Local+', 'MPDA-', 'MPDA']
        # 创建一个工作簿和工作表
        wb = Workbook()
        ws = wb.active
        
        # 设置列标题
        ws.append(names)
        
        # 将数据写入表格
        for user_id, selected_users, train_samples, test_samples, cloud, local, local_plus, mpda_minus, mpda in zip(user_id_list, num_selected_users_list, 
        num_train_samples_list, num_test_samples_list, cloud_list, local_list, local_plus_list, mpda_minus_list, mpda_list):
            ws.append([user_id, selected_users, train_samples, test_samples, cloud, local, local_plus, mpda_minus, mpda])
        
        # 保存 Excel 文件
        file_path = os.path.join(log_fp, str(task_index) + '.xlsx')
        wb.save(file_path)
        print(f"文件已保存为 {file_path}")
        return None
