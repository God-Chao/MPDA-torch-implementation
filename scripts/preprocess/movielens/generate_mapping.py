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
from utils.utils import utils, MovieLensDataset

# 读取数据
data = utils.get_raw_ratings()
print(f'raw ratings data has been loaded')

user_encoder = LabelEncoder()
data['user'] = user_encoder.fit_transform(data['user_id'])

item_encoder = LabelEncoder()
data['item'] = item_encoder.fit_transform(data['item_id'])

# 编码映射关系
user_mapping = dict(zip(user_encoder.classes_, user_encoder.transform(user_encoder.classes_)))
item_mapping = dict(zip(item_encoder.classes_, item_encoder.transform(item_encoder.classes_)))


user_mapping = {int(k): int(v) for k, v in user_mapping.items()}
user_path = os.path.join(utils.get_movielens_data_path(), 'user_mapping.json')
with open(user_path, "w") as json_file:
    json.dump(user_mapping, json_file)
print(f'user mapping json has been saved, path = {user_path}')


item_mapping = {int(k): int(v) for k, v in item_mapping.items()}
item_path = os.path.join(utils.get_movielens_data_path(), 'item_mapping.json')
# 保存为 JSON 文件
with open(item_path, "w") as json_file:
    json.dump(item_mapping, json_file)
print(f'item mapping json has been saved, path = {item_path}')
