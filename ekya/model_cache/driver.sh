
#!/bin/bash
# Runs one city at a time through ekya. Good as a sanity check to ensure retraining results in accuracy gains.
set -e

# CityScapes cities we are interested in:
# aachen bochum bremen cologne darmstadt dusseldorf jena monchengladbach
# stuttgart tubingen zurich
NUM_TASKS=10
# DATASET_NAME='cityscapes'
DATASET_NAME='waymo'
MODEL_NAME='resnet18'
HYPS_PATH='./hyp_map_all.json'
if [ ${DATASET_NAME} = 'cityscapes' ]
then
    echo "cityscapes"
    DATASET_PATH='../../datasets/CityScapes'
    CITIES=" bochum bremen cologne darmstadt dusseldorf jena monchengladbach stuttgart tubingen zurich"
    NO_TRAIN_MODEL_PATH='../../pretrained_models/pretrained_cityscapes_fftmunster_'${MODEL_NAME}'_1024x2.pt'

    SAVE_PATH='./results_golden_label_cityscapes'
elif [ ${DATASET_NAME} = 'waymo' ]
then
    echo 'waymo'
    DATASET_PATH='../../datasets/waymo/waymo_classification_images'
    CITIES="sf_000_009 sf_020_029 sf_030_039 sf_050_059 sf_060_069 sf_070_079 sf_080_089"

    SAVE_PATH='./results_golden_label_waymo'
    NO_TRAIN_MODEL_PATH='../../pretrained_models/waymo/waymo_resnet18_64.pth'
else
    echo 'Dataset type not supported!'
fi

echo ${DATASET_PATH}

# train customize model
# for CITY in ${CITIES}; do
#     echo ${CITY}
#     CUDA_VISIBLE_DEVICES=1 python reuse_model.py \
#         --dataset ${DATASET_NAME} \
#         --root ${DATASET_PATH} \
#         --camera_name ${CITY} \
#         --num_tasks ${NUM_TASKS} \
#         --checkpoint_path ${SAVE_PATH} \
#         --no_train_model_path ${NO_TRAIN_MODEL_PATH} \
#         --hyp_map_path ${HYPS_PATH}
# done

for CITY in ${CITIES}; do
    echo ${CITY}
    CUDA_VISIBLE_DEVICES=2 python run.py \
        --dataset ${DATASET_NAME} \
        --root ${DATASET_PATH} \
        --camera_name ${CITY} \
        --num_tasks ${NUM_TASKS} \
        --checkpoint_path ${SAVE_PATH} \
        --no_train_model_path ${NO_TRAIN_MODEL_PATH} \
        --hyp_map_path ${HYPS_PATH}
done
