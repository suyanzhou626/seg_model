#This script is used to run local test on car_damage. Users could also
# modify from this script for their use case.
#
# Usage:
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e
export ENVFILE=$1
source $ENVFILE
CURRENT_DIR=$(pwd)
WORK_DIR=${CURRENT_DIR}
SAVE_DIR=${RESULT_DIR}

LOG_DIR=$SAVE_DIR/$DATASET/$BACKBONE/log
mkdir -p $LOG_DIR
now=$(date +"i%Y%m%d_%H%M%S")
LOGNAME=$LOG_DIR/train-$LR-$BATCH_SIZE-$EPOCH-$INPUT_SIZE-$TEST_SIZE-$now.log

python -u ${WORK_DIR}/train.py \
--backbone $BACKBONE \
--dataset $DATASET \
--data_dir $DATA_DIR \
--train_list ${TRAIN_LIST} \
--val_list ${VAL_LIST} \
--input_size $INPUT_SIZE \
--test_size $TEST_SIZE \
--num_classes $NUM_CLASSES \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--save_dir $SAVE_DIR \
--lr $LR \
--shrink $SHRINK \
--normal_mean 0 0 0 \
--normal_std 255 \
2>&1|tee $LOGNAME