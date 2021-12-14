#!/bin/bash
# Runs one city at a time through ekya. Good as a sanity check to ensure retraining results in accuracy gains.
set -e

DATASET_PATH='/home/researcher/datasets/cityscapes/'
MODEL_PATH='/home/researcher/models/'
#DATASET_PATH='/home/romilb/datasets/cityscapes_raw/'
#MODEL_PATH='/home/romilb/research/msr/models/'
UTILITYSIM_HYPS_PATH='utilitysim_schedules/3city_0713_sysprof_fixedseed/hyp_map.json'
INFERENCE_PROFILE_PATH='real_inference_profiles.csv'
#UTILITYSIM_SCHEDULE_KEY='100_1_thief_True'
RETRAINING_PERIOD=200
NUM_TASKS=10
INFERENCE_CHUNKS=10
NUM_GPUS=1
EPOCHS=1  # Doesn't matter, read from utilitysim schedule.
START_TASK=1
TERMINATION_TASK=9
MAX_INFERENCE_RESOURCES=0.25

# Reduce number of epochs when using past retraining data
for UTILITYSIM_SCHEDULE_KEY in ${RETRAINING_PERIOD}_1_thief_True ${RETRAINING_PERIOD}_1_fair_dumb_True ${RETRAINING_PERIOD}_1_inference_only_True; do
  for CITIES in jena jena,zurich; do #jena,zurich,darmstadt,stuttgart,tubingen; do
    echo Running scheduler $UTILITYSIM_SCHEDULE_KEY on cities $CITIES
    NUM_CITIES=$(echo "${CITIES}" | awk -F "," ' { print NF } ')
    UTILITYSIM_SCHEDULE_PATH="utilitysim_schedules/cityscapes/0803_CacheFixed/data_eval_multicam_allocationdata_cityscapes_histacc_${NUM_CITIES}cities.json"
    echo $UTILITYSIM_SCHEDULE_PATH
    python driver_multicity.py --scheduler utilitysim \
             --cities ${CITIES} \
             --log-dir /tmp/ekya_expts/multicity/${NUM_CITIES}cities/${UTILITYSIM_SCHEDULE_KEY}/ \
             --retraining-period ${RETRAINING_PERIOD} \
             --num-tasks ${NUM_TASKS} \
             --inference-chunks ${INFERENCE_CHUNKS} \
             --num-gpus ${NUM_GPUS} \
             --root ${DATASET_PATH} \
             --use-data-cache \
             --restore-path ${MODEL_PATH} \
             --lists-pretrained frankfurt,munster \
             --hyperparameter-id 0 \
             --start-task ${START_TASK} \
             --termination-task ${TERMINATION_TASK} \
             --epochs ${EPOCHS} \
             --utilitysim-schedule-path ${UTILITYSIM_SCHEDULE_PATH} \
             --hyps-path ${UTILITYSIM_HYPS_PATH} \
             --utilitysim-schedule-key ${UTILITYSIM_SCHEDULE_KEY} \
             --inference-profile-path ${INFERENCE_PROFILE_PATH} \
             --max-inference-resources ${MAX_INFERENCE_RESOURCES}
  done
done