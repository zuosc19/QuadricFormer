PY_CONFIG=$1
WORK_DIR=$2

DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 63545";

echo "command = [torchrun $DISTRIBUTED_ARGS train.py]"
torchrun $DISTRIBUTED_ARGS train.py \
    --py-config $PY_CONFIG \
    --work-dir $WORK_DIR