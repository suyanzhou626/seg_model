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
SAVE_DIR=${RESULT_DIR}/${DATASET}/${BACKBONE}/${EXPERIMENT}/vis
RESUME=${RESULT_DIR}/${DATASET}/${BACKBONE}/${EXPERIMENT}/best.pth.tar

LOG_DIR=$SAVE_DIR/log
mkdir -p $LOG_DIR
now=$(date +"i%Y%m%d_%H%M%S")

python -u ${WORK_DIR}/vis.py \
--backbone $BACKBONE \
--data_dir $DATA_DIR \
--num_classes $NUM_CLASSES \
--batch_size 1 \
--save_dir $SAVE_DIR \
--val_list $VAL_LIST \
--resume $RESUME \
--test_size $TEST_SIZE \
--shrink $SHRINK \
--normal_mean 0 0 0 \
--normal_std 255 \
2>&1|tee $LOG_DIR/vis-$now.log