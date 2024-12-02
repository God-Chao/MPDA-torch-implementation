# Implementation MPDA with pytorch

## Data Preprocess
generate raw data for all users on device
```shell
nohup python -u ./scripts/preprocess/movielens/generate_device_data.py > ./log/generate_device_data.log 2>&1 &
```

generate seq data for all users on device
```shell
nohup python -u ./scripts/preprocess/movielens/generate_seq_data.py > ./log/generate_seq_data.log 2>&1 &
```

split train and test data for all users on device
```shell
nohup python -u ./scripts/preprocess/movielens/split.py > ./log/split_train_test_data.log 2>&1 &
```

generate user and mapping file
```shell
python -u scripts/preprocess/movielens/generate_mapping.py 
```

generate all users id json file
```shell
python scripts/preprocess/movielens/generate_all_users_list.py
```

generate users with train json file
```shell
python scripts/preprocess/movielens/generate_users_with_train.py
```

generate recall item pairs
```shell
python scripts/preprocess/movielens/generate_item_pairs.py
```

## Initial Model
train global model NCF
```shell
nohup python -u ./model/NCF/train_global_model.py > ./log/train_global_model_NCF.log 2>&1 &
```

transfer model NCF
```shell
bash ./commands/ncf_movielens_50_random.sh
```