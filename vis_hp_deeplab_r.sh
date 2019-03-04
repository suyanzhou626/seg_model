EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/vis.py

DATA_DIR=/mnt/lustre/wuyao/Data/segmentation_data/humanparse_seg/human_parse_257
DATASETVAL=renren
DATASETTRAIN=renren
BACKBONE=deeplab
EXPERIMENT=experiment_0
SAVE_DIR=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/humanparse_257/$DATASETTRAIN/$BACKBONE/$EXPERIMENT/vis_val_$DATASETVAL
VIS_LIST=/mnt/lustre/wuyao/dataset_list/humanparse_257/${DATASETVAL}_val.txt
RESUME=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/humanparse_257/$DATASETTRAIN/$BACKBONE/$EXPERIMENT/best.pth.tar
NUM_CLASSES=2
BATCH_SIZE=1
CROP_SIZE=257
SHRINK=16

JOBNAME=vis_hp_renren
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