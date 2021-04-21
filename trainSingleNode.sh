
# /opt/conda/bin/python train_dirset.py 2>&1 | tee log_trainAtDocker.log &
# /opt/conda/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py 2>&1 | tee insightLog.txt &
# KMP_INIT_AT_FORK=FALSE /opt/conda/bin/python train_distributed.py 2>&1 | tee log_trainAtDocker.log &
NCCL_DEBUG=INFO /opt/conda/bin/python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py 2>&1 | tee insightSingleNode.txt &


