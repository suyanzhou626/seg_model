EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/test_video.py

TEST_PATH=/mnt/lustre/wuyao/Data/video/fore_back/test
BACKBONE=vnet
EXPERIMENT=experiment_0
DATASET_SAVE=video_fore_back
DATASET_TRAIN=renren
SAVE_DIR=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/humanparse_720/$DATASET_SAVE/$BACKBONE/test_${DATASET_TRAIN}_${EXPERIMENT}_post
RESUME=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/humanparse_720/$DATASET_TRAIN/$BACKBONE/$EXPERIMENT/best.pth.tar
NUM_CLASSES=2
BATCH_SIZE=1
CROP_SIZE=737
SHRINK=32

JOBNAME=video_test
#LOG_DIR=$SAVE_DIR
#mkdir -p $LOG_DIR
part=HA_senseAR
now=$(date +"i%Y%m%d_%H%M%S")

srun --partition=$part --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 -n1 --job-name=$JOBNAME python \
-u $EXECUTE_DIR \
--backbone $BACKBONE \
--test_path $TEST_PATH \
--num_classes $NUM_CLASSES \
--save_dir $SAVE_DIR \
--resume $RESUME \
--test_size $CROP_SIZE \
--shrink $SHRINK \
--bgr_mode \
--blursize 3 \
--diff_threshold 0.9 \
--hole_ratio 0.05 \
