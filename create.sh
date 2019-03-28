set -e
export ENVFILE=$1
source $ENVFILE
rm -rf $DATA_DIR
mkdir -p $DATA_DIR
python -u json_pre.py \
--out_path $DATA_DIR \
--aug_times $AUG_TIMES \
--mode $MODE \
--read_path $ORIGIN_DIR \
--config $CONFIG \