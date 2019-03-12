EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/link_train.py

DATA_DIR=/mnt
SAVE_DIR=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/pascal
DATASET=all
TRAIN_LIST=/mnt/lustre/wuyao/dataset_list/voc2012/${DATASET}_train.txt
VAL_LIST=/mnt/lustre/wuyao/dataset_list/voc2012/${DATASET}_val.txt 
BACKBONE=deeplab_gai
CROP_SIZE=513
TEST_SIZE=513
NUM_CLASSES=21
EPOCH=1000
BATCH_SIZE=4
LR=0.01
SHRINK=16

NORMAL_STD=255
ROTATE=30
LOSS_TYPE=focal
DISPLAY_ITER=100
LR_SCHEDULER=step
MOMENTUM=0.8
WEIGHT_DECAY=0.001
RESUME=$SAVE_DIR/$DATASET/$BACKBONE/experiment_0/checkpoint.pth.tar

JOBNAME=deplab_pa
LOG_DIR=$SAVE_DIR/$DATASET/$BACKBONE/log
mkdir -p $LOG_DIR
part=HA_senseAR
now=$(date +"i%Y%m%d_%H%M%S")
LOGNAME=$LOG_DIR/train-$LR-$BATCH_SIZE-$EPOCH-$CROP_SIZE-$TEST_SIZE-$now.log

srun --partition=$part --mpi=pmi2 --gres=gpu:8 --ntasks-per-node=8 -n8 --job-name=$JOBNAME -x SH-IDC1-10-5-36-191 python \
-u $EXECUTE_DIR \
--backbone $BACKBONE \
--dataset $DATASET \
--data_dir $DATA_DIR \
--train_list $TRAIN_LIST \
--val_list $VAL_LIST \
--crop_size $CROP_SIZE \
--test_size $TEST_SIZE \
--num_classes $NUM_CLASSES \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--save_dir $SAVE_DIR \
--lr $LR \
--shrink $SHRINK \
--ft \
2>&1|tee $LOGNAME
