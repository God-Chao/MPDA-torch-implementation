import pandas as pd
import json
import os
import sys
import random

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

def recall_item_pairs(data, num_pairs):
    # 统计每个物品的交互次数
    item_popularity = data['item_id'].value_counts()
    # 设置流行度阈值
    threshold = item_popularity.median()
    # 划分流行和不流行物品
    popular_items = item_popularity[item_popularity > threshold].index.tolist()
    non_popular_items = item_popularity[item_popularity <= threshold].index.tolist()

    # 生成物品对
    popular_pairs = [
        tuple(random.sample(popular_items, 2)) for _ in range(num_pairs // 2)
    ]
    non_popular_pairs = [
        tuple(random.sample(non_popular_items, 2)) for _ in range(num_pairs // 2)
    ]

    # 合并并返回
    return popular_pairs + non_popular_pairs

data = utils.get_raw_ratings()
num_recall_item_pairs = utils.get_num_recall_item_pairs()

recall_item_pairs = recall_item_pairs(data, num_recall_item_pairs)

# 保存到json文件中
file_path = os.path.join(utils.get_movielens_data_path(), 'recall_item_pairs.json')

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(recall_item_pairs, f, indent=4, ensure_ascii=False)


# 保存为 JSON 格式
print(f"物品对已保存为 JSON 文件: {file_path}")