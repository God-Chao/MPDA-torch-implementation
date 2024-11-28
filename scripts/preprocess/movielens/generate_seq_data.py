import argparse
import os
import pandas as pd
import numpy as np

def config_args():
    parser = argparse.ArgumentParser('generate seq data for all users')

    parser.add_argument('-processed_fp', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/processed', help='所有用户目录')

    args = parser.parse_args()
    return args

# 将input_path的csv文件转换为顺序递增的形式并输出到output_path中
def convert_interaction_records(input_path, output_path):
    # 读取数据
    data = pd.read_csv(input_path, header=0, names=['user_id', 'movie_id', 'rating', 'timestamp'])

    # 初始化历史记录
    history_ids = []
    history_ratings = []

    # 用于存储结果的列表
    converted_data = []

    # 遍历每一条记录
    for _, row in data.iterrows():
        # 获取当前的用户、电影、评分和时间戳
        user_id = int(row['user_id'])
        movie_id = int(row['movie_id'])
        rating = row['rating']
        timestamp = int(row['timestamp'])

        # 生成转换后的行
        converted_data.append([
            user_id,
            movie_id,
            ' '.join(map(str, history_ids)) if history_ids else '',
            ' '.join(map(str, history_ratings)) if history_ratings else '',
            rating,
            timestamp
        ])

        # 更新历史记录
        history_ids.append(movie_id)
        history_ratings.append(rating)

    # 将结果转换为 DataFrame
    converted_df = pd.DataFrame(
        converted_data, columns=['user_id', 'movie_id', 'history_ids', 'history_ratings', 'rating', 'timestamp']
    )

    # 保存结果
    converted_df.to_csv(output_path, index=False, header=False)

def main():
    args = config_args()
    # 遍历所有用户生成seq文件
    input_file = ''
    output_file = ''

    for user in os.listdir(args.processed_fp):
        user_path = os.path.join(args.processed_fp, user)
        if os.path.isdir(user_path):  # 检查是否为目录
            input_file = os.path.join(user_path, 'user_'+user+'_raw_data.csv')
            output_file = os.path.join(user_path, 'user_'+user+'_seq.csv')
            convert_interaction_records(input_file, output_file)
        print(f'user{user}/{len(os.listdir(args.processed_fp))} seq data has been generated')


if __name__ == '__main__':
    # 利用每个用户的raw_data文件生成seq文件，以便后续模型的训练
    print(f'start generate seq data for all users')
    main()
    print('all users has been examined!')