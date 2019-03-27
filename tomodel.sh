set -e
export ENVFILE=$1
source $ENVFILE
CURRENT_DIR=$(pwd)
WORK_DIR=${CURRENT_DIR}
SAVE_DIR=${RESULT_DIR}/${DATASET}/${BACKBONE}/${EXPERIMENT}
RESUME=${RESULT_DIR}/${DATASET}/${BACKBONE}/${EXPERIMENT}/best.pth.tar

python -u ${WORK_DIR}/tomodel.py \
--backbone $BACKBONE \
--dataset $DATASET \
--num_classes $NUM_CLASSES \
--save_dir $SAVE_DIR \
--resume $RESUME \