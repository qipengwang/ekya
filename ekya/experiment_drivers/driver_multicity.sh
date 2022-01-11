#!/bin/bash
# Runs multiple cities city at a time through ekya. Good as a sanity check to ensure retraining results in accuracy gains.
set -e

DATASET_PATH='/home/researcher/datasets/cityscapes/'
MODEL_PATH='/home/researcher/models/'
INFERENCE_PROFILE_PATH='real_inference_profiles.csv'
RETRAINING_PERIOD=100
NUM_TASKS=10
INFERENCE_CHUNKS=10
NUM_GPUS=1
EPOCHS=1
START_TASK=1
TERMINATION_TASK=8
MAX_INFERENCE_RESOURCES=0.25
CITIES=zurich,jena,cologne

# Reduce number of epochs when using past retraining data
for scheduler in utilitysim fair noretrain thief; do
  echo Running scheduler $scheduler on cities $CITIES
  python driver_multicity.py --scheduler ${scheduler} \
           --cities ${CITIES} \
           --log-dir /tmp/ekya_expts/multicity/${scheduler}/ \
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
           --inference-profile-path ${INFERENCE_PROFILE_PATH} \
           --max-inference-resources ${MAX_INFERENCE_RESOURCES}
done