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

trainset_path = os.path.join(utils.get_movielens_data_path(), 'trainset')

# 读取数据该目录下所有用户
# 提取用户 ID
user_ids = []
for file in os.listdir(trainset_path):
    if file.endswith(".csv") and file.split(".")[0].isdigit():
        user_ids.append(int(file.split(".")[0]))

# 保存为json文件
output_json = os.path.join(utils.get_movielens_data_path(), 'user_with_train.json')
with open(output_json, "w") as json_file:
    json.dump(user_ids, json_file, indent=4)
print(f"User IDs successfully saved to {output_json}")

