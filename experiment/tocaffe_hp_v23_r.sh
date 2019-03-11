EXECUTE_DIR=/mnt/lustre/wuyao/seg_model/tocaffe.py

DATASETTRAIN=renren
BACKBONE=v23_4x
EXPERIMENT=experiment_0
NUM_CLASSES=2
INPUT_SHAPE_CHA=3
INPUT_SHAPE_HEIGHT=257
INPUT_SHAPE_WIDTH=257
MODEL_NAME=M_Segment_V23_Human_${DATASETTRAIN}_$EXPERIMENT

SAVE_DIR=/mnt/lustre/wuyao/Data/tocaffe_model/$MODEL_NAME/cnn
RESUME=/mnt/lustre/wuyao/Data/segmentation_pytorch_model/humanparse_257/$DATASETTRAIN/$BACKBONE/$EXPERIMENT/best.pth.tar

mkdir -p ${SAVE_DIR}
JOBNAME=tocaffe_h
part=HA_senseAR

srun --partition=$part --mpi=pmi2 --gres=gpu:1 --ntasks-per-node=1 -n1 --job-name=$JOBNAME python \
-u $EXECUTE_DIR \
--backbone $BACKBONE \
--num_classes $NUM_CLASSES \
--input_shape $INPUT_SHAPE_CHA $INPUT_SHAPE_HEIGHT $INPUT_SHAPE_WIDTH \
--save_dir ${SAVE_DIR} \
--resume $RESUME \

python -m nart_tools.caffe.convert --bns -B ${SAVE_DIR}/model.prototxt ${SAVE_DIR}/model.caffemodel
mv ${SAVE_DIR}/model-convert.prototxt ${SAVE_DIR}/rel.prototxt
mv ${SAVE_DIR}/model-convert.caffemodel ${SAVE_DIR}/model.bin
#rm ${SAVE_DIR}/model.prototxt
#rm ${SAVE_DIR}/model.proto
#rm ${SAVE_DIR}/model.caffemodel
