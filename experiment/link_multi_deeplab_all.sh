EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/link_train.py

DATA_DIR=/mnt/lustre/wuyao/Data/segmentation_data/multiclass_seg/data481
SAVE_DIR=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/multiclass_481
DATASET=all
TRAIN_LIST=/mnt/lustre/wuyao/dataset_list/multiclass/${DATASET}_train.txt
VAL_LIST=/mnt/lustre/wuyao/dataset_list/multiclass/${DATASET}_val.txt 
BACKBONE=deeplabv3plus
CROP_SIZE=289
TEST_SIZE=481
NUM_CLASSES=15
EPOCH=1000
BATCH_SIZE=16
LR=0.1
SHRINK=16

NORMAL_STD=255
ROTATE=45
LOSS_TYPE=focal
DISPLAY_ITER=100
LR_SCHEDULER=step
MOMENTUM=0.8
WEIGHT_DECAY=0.001
RESUME=$SAVE_DIR/$DATASET/$BACKBONE/experiment_0/checkpoint.pth.tar

JOBNAME=m_dpv3+_all
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
--input_size $CROP_SIZE \
--test_size $TEST_SIZE \
--num_classes $NUM_CLASSES \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--save_dir $SAVE_DIR \
--lr $LR \
--shrink $SHRINK \
--bgr_mode \
2>&1|tee $LOGNAME
