EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/test_video.py

TEST_PATH=/mnt/lustre/wuyao/Data/multi_video/test
BACKBONE=vnetprun2
EXPERIMENT=experiment_0
DATASET_SAVE=multi_video
DATASET_TRAIN=all
SAVE_DIR=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/multiclass_481/$DATASET_SAVE/$BACKBONE/$EXPERIMENT/test_$DATASET_TRAIN
RESUME=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/multiclass_481/$DATASET_TRAIN/$BACKBONE/$EXPERIMENT/best.pth.tar
NUM_CLASSES=15
BATCH_SIZE=1
CROP_SIZE=481
SHRINK=32

JOBNAME=video_test
LOG_DIR=$SAVE_DIR/log
mkdir -p $LOG_DIR
part=HA_senseAR
now=$(date +"i%Y%m%d_%H%M%S")

srun --partition=$part --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 -n1 --job-name=$JOBNAME python \
-u $EXECUTE_DIR \
--backbone $BACKBONE \
--test_path $TEST_PATH \
--num_classes $NUM_CLASSES \
--save_dir $SAVE_DIR \
--resume $RESUME \
--crop_size $CROP_SIZE \
--shrink $SHRINK \
2>&1|tee $LOG_DIR/video-vis-$now.log
