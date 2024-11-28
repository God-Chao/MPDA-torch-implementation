import argparse
import os
import pandas as pd

def config_args():
    parser = argparse.ArgumentParser('generate raw data for all users')

    parser.add_argument('-raw_data_fp', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/ratings.csv', help='原始Movielens-20m数据集的ratings.csv文件')

    parser.add_argument('-target_dir_path', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/processed', help='目标目录路径')

    args = parser.parse_args()
    return args


def main():
    args = config_args()
    raw_data = pd.read_csv(args.raw_data_fp)
    num_users = 138493

    os.makedirs(args.target_dir_path, exist_ok=True)  # 创建端侧目录

    # 按用户分组并写入文件
    for user_id, group in raw_data.groupby('userId'):
        # 按时间排序
        group = group.sort_values(by='timestamp')
        # 为每个用户创建目录
        user_dir = os.path.join(args.target_dir_path, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        
        # 写入用户数据
        user_file = os.path.join(user_dir, f'user_{user_id}_raw_data.csv')
        group.to_csv(user_file, index=False)
        print(f"User {user_id}/{num_users} data saved to {user_file}")


if __name__ == '__main__':
    # 将MovieLens数据集中每个用户创建一个自己的数据集作为端侧本地数据
    print(f'start generate on device raw data for all users')
    main()
    print('all users has been generated!')