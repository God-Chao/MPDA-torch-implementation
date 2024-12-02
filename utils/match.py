import random
from .utils import utils
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 云端召回算法
class match():
    # 随机匹配
    def random_match(k):
        all_users = utils.get_user_with_train()
        selected_users = random.sample(all_users, k)
        return selected_users
    
    # 根据用户交互历史的相似度匹配
    def match_by_interaction(k):
        random.seed(utils.get_random_seed())
        all_users = utils.get_all_users()

        # 计算当前用户与所有用户的余弦相似度
        user = user.reshape(1, -1)  # 确保用户交互历史为二维数组
        similarities = cosine_similarity(user, all_users).flatten()
        
        # 排除自身相似度（假设用户自己在 all_users 中的位置为 0）
        similarities[0] = -np.inf
        
        # 获取相似度最高的 k 个用户的索引
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # 返回相似用户索引及相似度
        return [(idx, similarities[idx]) for idx in top_k_indices]

    # def recall_item_similarity(k):
        
