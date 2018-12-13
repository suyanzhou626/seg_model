EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/train.py
DATA_DIR=/mnt/lustrenew/liutinghao/train_data/data/segmentation_data/multiclass_seg/data257
SAVE_DIR=/mnt/lustre/wuyao/segmentation/muticlass_257
DATASET=all
TRAIN_LIST=/mnt/lustre/wuyao/muticlass_all_train.txt
VAL_LIST=/mnt/lustre/wuyao/muticlass_all_val.txt
BACKBONE=dbl
CROP_SIZE=225
NUM_CLASSES=15
EPOCH=200
BATCH_SIZE=256

LOG_DIR=$SAVE_DIR/$DATASET/$BACKBONE/log
mkdir -p $LOG_DIR
part=Pose
now=$(date +"i%Y%m%d_%H%M%S")

srun --partition=Pose --mpi=pmi2 --gres=gpu:8 --ntasks-per-node=1 -n1 --job-name=multiclass_seg python -u \
$EXECUTE_DIR --backbone $BACKBONE --dataset $DATASET --data_dir $DATA_DIR --crop_size $CROP_SIZE --num_classes $NUM_CLASSES \
--epoch $EPOCH --batch_size $BATCH_SIZE \
--use_balanced_weights --save_dir $SAVE_DIR --lr 0.1 --train_list $TRAIN_LIST --val_list $VAL_LIST 2>&1|tee $LOG_DIR/train-$now.log