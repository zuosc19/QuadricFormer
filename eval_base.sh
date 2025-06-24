torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 63546 eval.py \
    --py-config config/nusc_surroundocc_sq12800.py \
    --work-dir work_dir/sq12800/debug
