EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/train.py
DATA_DIR=/mnt/lustrenew/liutinghao/train_data/data/segmentation_data/hair_seg/datasets
SAVE_DIR=/mnt/lustre/wuyao/segmentation/hair_seg
DATASET=zhibo_selfie_rgbd
BACKBONE=v23_4x
CROP_SIZE=225
NUM_CLASSES=2
EPOCH=500
BATCH_SIZE=256

LOG_DIR=$SAVE_DIR/$DATASET/$BACKBONE/log
mkdir -p $LOG_DIR
part=Pose
now=$(date +"i%Y%m%d_%H%M%S")

srun --partition=Pose --mpi=pmi2 --gres=gpu:4 --ntasks-per-node=1 -n1 --job-name=hair_v5 python \
$EXECUTE_DIR --backbone $BACKBONE --dataset $DATASET --data_dir $DATA_DIR --crop_size $CROP_SIZE --num_classes $NUM_CLASSES \
--epoch $EPOCH --batch_size $BATCH_SIZE \
--use_balanced_weights --save_dir $SAVE_DIR --lr 0.1 2>&1|tee $LOG_DIR/train-$now.log