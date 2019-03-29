EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/vis.py

DATA_DIR=/mnt/lustre/wuyao/Data/segmentation_data/multiclass_seg/data481
DATASETVAL=all
DATASETTRAIN=all
BACKBONE=vnetprun3
EXPERIMENT=experiment_0
SAVE_DIR=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/multiclass_481/$DATASETTRAIN/$BACKBONE/$EXPERIMENT/vis_val_$DATASETVAL
VIS_LIST=/mnt/lustre/wuyao/dataset_list/multiclass/${DATASETVAL}_val.txt
RESUME=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/multiclass_481/$DATASETTRAIN/$BACKBONE/$EXPERIMENT/best.pth.tar
NUM_CLASSES=15
BATCH_SIZE=1
CROP_SIZE=481
SHRINK=32

JOBNAME=vis_multi_all
LOG_DIR=$SAVE_DIR/log
mkdir -p $LOG_DIR
part=HA_senseAR
now=$(date +"i%Y%m%d_%H%M%S")

srun --partition=$part --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 -n1 --job-name=$JOBNAME python \
-u $EXECUTE_DIR \
--backbone $BACKBONE \
--data_dir $DATA_DIR \
--num_classes $NUM_CLASSES \
--batch_size $BATCH_SIZE \
--save_dir $SAVE_DIR \
--vis_list $VIS_LIST \
--resume $RESUME \
--crop_size $CROP_SIZE \
--shrink $SHRINK \
2>&1|tee $LOG_DIR/vis-$now.log
