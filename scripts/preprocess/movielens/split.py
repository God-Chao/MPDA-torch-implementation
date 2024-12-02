import argparse
import os
import pandas as pd
import json

def config_args():
    parser = argparse.ArgumentParser('generate seq data for all users')

    parser.add_argument('-processed_fp', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/processed', help='所有用户目录')
    parser.add_argument('-trainset_fp', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/trainset', help='训练集目录')
    parser.add_argument('-testset_fp', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/testset', help='测试集目录')
    parser.add_argument('-timestamp', type=int, default=1225642324, help='划分的时间戳依据')
    parser.add_argument('-user_intersect_json_fp', type=str, default='/home/chao/workspace/MPDA-implementation/data/MovieLens/ml-20m/user_intersect.json', help='既有训练集又有测试集的用户json路径')


    args = parser.parse_args()
    return args


def main():
    args = config_args()
    # 划分训练集和测试集，timestamp在1225642324之前的为训练集，之后的为测试集
    data = None
    users_with_files = []

    # 检查训练集目录和测试集目录是否存在，不存在则创建
    if not os.path.exists(args.trainset_fp):
        os.makedirs(args.trainset_fp)
        print('trainset_fp has been created')
    if not os.path.exists(args.testset_fp):
        os.makedirs(args.testset_fp)
        print('testset_fp has been created')
    
    for user in os.listdir(args.processed_fp):
        user_path = os.path.join(args.processed_fp, user)
        if os.path.isdir(user_path):  # 检查是否为目录
            data_path = os.path.join(user_path, 'user_'+user+'_seq.csv')
            data = pd.read_csv(data_path, names=["user_id", "movie_id", "history_ids", "history_ratings", "rating", "timestamp"])
            # 将 timestamp 转换为整数（如果不是整数类型）
            data["timestamp"] = data["timestamp"].astype(int)

            # 定义时间戳的分界线
            split_timestamp = args.timestamp

            # 按时间戳划分数据
            train_data = data[data["timestamp"] < split_timestamp]
            test_data = data[data["timestamp"] >= split_timestamp]

            # 保存到 CSV 文件
            if len(train_data) > 0:
                train_data.to_csv(args.trainset_fp +'/'+ user + '.csv', index=False, header=False)
                print(f'user{user}/{len(os.listdir(args.processed_fp))} train data has been generated')
            if len(test_data) > 0:
                test_data.to_csv(args.testset_fp +'/'+ user + '.csv', index=False, header=False)
                print(f'user{user}/{len(os.listdir(args.processed_fp))} test data has been generated')
            if len(train_data) > 0 and len(test_data) > 0:
                users_with_files.append(user)
                print(f'user{user}/{len(os.listdir(args.processed_fp))} has been wirtten to user_intersect.json')

    print(f'start write user_intersect.json')
     # 将符合条件的用户写入 JSON 文件
    with open(args.user_intersect_json_fp, 'w') as json_file:
        json.dump(users_with_files, json_file, indent=4)
    print(f'user_intersect.json has finished')
    print('all users has been splitted!')
    

if __name__ == '__main__':
    # 将每个用户的seq文件按时间戳划分训练集和测试集
    print(f'start split train and test data for all users')
    main()
    print('all files has been generated')
    