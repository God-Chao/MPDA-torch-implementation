import torch
import torch.nn as nn


# 定义NCF模型
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, dropout):
        super(NCF, self).__init__()
        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP层
        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64), # 将用户和物品嵌入拼接作为输入
            nn.ReLU(),
            nn.Dropout(dropout), # 随机丢弃一部分神经元，防止过拟合

            nn.Linear(64, 32), # 将用户和物品嵌入拼接作为输入
            nn.ReLU(),
            nn.Dropout(dropout) # 随机丢弃一部分神经元，防止过拟合
        )

        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid() # 将输出值压缩到[0, 1]之间，用于概率预测
        )
        
    def forward(self, user_indices, item_indices):
        # 获取嵌入向量
        user_vector = self.user_embedding(user_indices)
        item_vector = self.item_embedding(item_indices)
        
        # 拼接用户和物品向量
        vector = torch.cat([user_vector, item_vector], dim=-1)
        
        # MLP层
        output = self.mlp_layers(vector)

        # 输出层
        predict = self.output_layers(output)

        return predict
    