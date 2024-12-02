import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import os
import sys

# 当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 当前文件的父目录（需要调整路径的深度根据你的项目结构而定）
parent_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
# 将目标目录加入到 sys.path
sys.path.append(parent_dir)
from utils.utils import utils

# 读取数据
data = utils.get_raw_ratings()
print(f'raw ratings data has been loaded')

all_users = data['user_id'].unique()
user_path = os.path.join(utils.get_movielens_data_path(), 'all_users_id.json')
with open(user_path, 'w') as json_file:
    json.dump(all_users.tolist(), json_file, indent=4)
print(f'all users id json has been saved, path = {user_path}')

