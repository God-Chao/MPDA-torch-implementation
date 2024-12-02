home_path="/home/chao/workspace/MPDA-implementation"

run_name="transfer_movielens_ncf_50_random"

mkdir -p "${home_path}/log/${run_name}/running_logs"

recall_num=50
recall_alg="random"

for ti in $(seq 0 1 9)
do
  nohup python -u ${home_path}/model/NCF/transfer.py -task_index=${ti} -recall_num=${recall_num} -recall_alg=${recall_alg} > ${home_path}/log/${run_name}/running_logs/${ti}.log 2>&1 &
done