PY_CONFIG=$1
CKPT_PATH=$2
SCENE_NAME=$3
WORK_DIR=$4

export QT_QPA_PLATFORM=offscreen
python -m torch.distributed.launch --nproc_per_node=1 --master_port=63545 \
    --use_env vis.py \
    --py-config $PY_CONFIG \
    --work-dir $WORK_DIR \
    --load-from $CKPT_PATH \
    --vis_occ \
    --scene-name $SCENE_NAME