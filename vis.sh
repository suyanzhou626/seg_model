EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/vis.py
DATA_DIR=/mnt/lustrenew/liutinghao/train_data/data/segmentation_data/hair_seg/datasets
DATASET=zhibo_selfie_rgbd
BACKBONE=v23_4x
SAVE_DIR=/mnt/lustre/wuyao/segmentation/hair_seg/$DATASET/$BACKBONE/vis_val
VIS_LIST=hair_val_list_$DATASET.txt
RESUME=/mnt/lustre/wuyao/segmentation/hair_seg/$DATASET/$BACKBONE/model_best.pth.tar
NUM_CLASSES=2
BATCH_SIZE=1

LOG_DIR=$SAVE_DIR/log
mkdir -p $LOG_DIR
part=Pose
now=$(date +"i%Y%m%d_%H%M%S")

srun --partition=Pose --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 -n1 --job-name=hair_vis python \
-u $EXECUTE_DIR --backbone $BACKBONE --data_dir $DATA_DIR --num_classes $NUM_CLASSES \
--batch_size $BATCH_SIZE --save_dir $SAVE_DIR --vis_list $VIS_LIST --resume $RESUME 2>&1|tee $LOG_DIR/vis-$now.log