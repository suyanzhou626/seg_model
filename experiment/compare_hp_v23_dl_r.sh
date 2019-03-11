EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/compare.py

DATA_DIR=/mnt/lustre/wuyao/Data/segmentation_data/humanparse_seg/human_parse_257
DATASETVAL=all
DATASETTRAIN=renren
BACKBONE1=v23_4x
BACKBONE2=deeplab
EXPERIMENT=experiment_0
SAVE_DIR=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/humanparse_257/$DATASETTRAIN/${BACKBONE1}_comp_${BACKBONE2}/$EXPERIMENT/vis_val_$DATASETVAL
VIS_LIST=/mnt/lustre/wuyao/dataset_list/humanparse_257/${DATASETVAL}_val.txt
RESUME1=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/humanparse_257/$DATASETTRAIN/$BACKBONE1/$EXPERIMENT/best.pth.tar
RESUME2=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/humanparse_257/$DATASETTRAIN/$BACKBONE2/$EXPERIMENT/best.pth.tar
NUM_CLASSES=2
BATCH_SIZE=1
CROP_SIZE=257
SHRINK=16

JOBNAME=comp_hp_renren
LOG_DIR=$SAVE_DIR/log
mkdir -p $LOG_DIR
part=HA_senseAR
now=$(date +"i%Y%m%d_%H%M%S")

srun --partition=$part --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 -n1 --job-name=$JOBNAME python \
-u $EXECUTE_DIR \
--backbone1 $BACKBONE1 \
--backbone2 $BACKBONE2 \
--data_dir $DATA_DIR \
--num_classes $NUM_CLASSES \
--batch_size $BATCH_SIZE \
--save_dir $SAVE_DIR \
--vis_list $VIS_LIST \
--resume1 $RESUME1 \
--resume2 $RESUME2 \
--crop_size $CROP_SIZE \
--shrink $SHRINK \
2>&1|tee $LOG_DIR/vis-$now.log
