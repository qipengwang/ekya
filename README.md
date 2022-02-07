# Ekya: Continuous Learning on the Edge
Ekya is a system which enables continuous learning on resource constrained devices.

![Ekya Architecture](https://i.imgur.com/ng1jLsS.png)

More details can be found in our NSDI 2022 paper available [here](https://nsdi22spring.usenix.hotcrp.com/doc/nsdi22spring-paper74.pdf). 

# Installation

1. First checkout Ray repository. Ekya requires first building a particular branch of Ray from source. Ekya uses commit `cf53b351471716e7bfa71d36368ebea9b0e219c5` (`Ray 0.9.0.dev0`) from the Ray repository.
`pip install ray` is not sufficient. 
```bash
git clone https://github.com/ray-project/ray/
cd ray
git checkout cf53b35
```
2. To build Ray, follow the [build instructions](https://docs.ray.io/en/master/development.html#building-ray-on-linux-macos-full) from the Ray repository.
```
sudo apt-get update
sudo apt-get install -y build-essential curl unzip psmisc

# Install Cython
pip install cython==0.29.0 pytest

# Install Bazel.
ray/ci/travis/install-bazel.sh
# (Windows users: please manually place Bazel in your PATH, and point
# BAZEL_SH to MSYS2's Bash: ``set BAZEL_SH=C:\Program Files\Git\bin\bash.exe``)

# Build the dashboard
# (requires Node.js, see https://nodejs.org/ for more information).
# If folder "ray/dasboard/client" does not exist, please move forward to 
# "Install ray"
pushd ray/dashboard/client
npm install
npm run build
popd

# Install Ray.
cd ray/python
pip install -e . --verbose  # Add --user if you see a permission denied error.
```
3. After installing ray, clone the Ekya repository and install Ekya.
```
git clone https://github.com/romilbhardwaj/ekya/
pip install -e . --verbose
``` 
4. Install [Nvidia Multiprocess Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html).
```
sudo apt-get update
sudo apt-get install nvidia-cuda-mps
```
5. Set your GPU to run in exclusive process mode and run Nvidia MPS daemon. This will require killing Xserver if it is running.
```
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi -i 2 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
```

# Running Ekya with Cityscapes Dataset
1. Setup the Cityscapes dataset using the instructions on the [website](https://www.cityscapes-dataset.com/) 
2. Download the pretrained models for cityscapes from [here](https://drive.google.com/drive/folders/15qE5IBFAkKuiDeUcV8xQPvpKXq1Zk6yT?usp=sharing).
3. Run the multicity training script is provided with Ekya.
 ```
./ekya/experiment_drivers/driver_multicity.sh
``` 
You may need to modify `DATASET_PATH` and `MODEL_PATH` to point to your dataset and pretrained models dir, respectively.
This script will run all schedulers, including `thief`, `fair` and `noretrain`.

## Ekya driver script usage guide
```
usage: Ekya  [-h] [-ld LOG_DIR] [-retp RETRAINING_PERIOD]
                    [-infc INFERENCE_CHUNKS] [-numgpus NUM_GPUS]
                    [-memgpu GPU_MEMORY] [-r ROOT]
                    [--dataset-name {cityscapes,waymo}] [-c CITIES]
                    [-lpt LISTS_PRETRAINED] [-lp LISTS_ROOT] [-dc]
                    [-ir RESIZE_RES] [-w NUM_WORKERS] [-ts TRAIN_SPLIT]
                    [-dtfs] [-hw HISTORY_WEIGHT] [-rp RESTORE_PATH]
                    [-cp CHECKPOINT_PATH] [-mn MODEL_NAME] [-nc NUM_CLASSES]
                    [-b BATCH_SIZE] [-lr LEARNING_RATE] [-mom MOMENTUM]
                    [-e EPOCHS] [-nh NUM_HIDDEN] [-dllo] [-sched SCHEDULER]
                    [-usp UTILITYSIM_SCHEDULE_PATH]
                    [-usc UTILITYSIM_SCHEDULE_KEY] [-mpd MICROPROFILE_DEVICE]
                    [-mprpt MICROPROFILE_RESOURCES_PER_TRIAL]
                    [-mpe MICROPROFILE_EPOCHS]
                    [-mpsr MICROPROFILE_SUBSAMPLE_RATE]
                    [-mpep MICROPROFILE_PROFILING_EPOCHS]
                    [-fswt FAIR_INFERENCE_WEIGHT] [-nt NUM_TASKS]
                    [-stid START_TASK] [-ttid TERMINATION_TASK]
                    [-nsp NUM_SUBPROFILES] [-op RESULTS_PATH] [-uhp HYPS_PATH]
                    [-hpid HYPERPARAMETER_ID] [-pm] [-pp PROFILE_WRITE_PATH]
                    [-ipp INFERENCE_PROFILE_PATH]
                    [-mir MAX_INFERENCE_RESOURCES]

Ekya driver script for cityscapes dataset. Uses pretrained models to improve accuracy over time.

optional arguments:
  -h, --help            show this help message and exit
  -ld LOG_DIR, --log-dir LOG_DIR
                        Directory to log results to
  -retp RETRAINING_PERIOD, --retraining-period RETRAINING_PERIOD
                        Retraining period in seconds
  -infc INFERENCE_CHUNKS, --inference-chunks INFERENCE_CHUNKS
                        Number of inference chunks per retraining window.
  -numgpus NUM_GPUS, --num-gpus NUM_GPUS
                        Number of GPUs to partition.
  -memgpu GPU_MEMORY, --gpu-memory GPU_MEMORY
                        Per GPU Memory in GB.
  -r ROOT, --root ROOT  Path to cityscapes dataset root.
  --dataset-name {cityscapes,waymo}
                        Name of the dataset supported.
  -c CITIES, --cities CITIES
                        comma separated str of list of cities to create
                        cameras. Num cameras = num of cities
  -lpt LISTS_PRETRAINED, --lists-pretrained LISTS_PRETRAINED
                        comma separated str of lists used for training the
                        pretrained model. Used as history for continuing the
                        retraining. Usually frankfurt,munster.
  -lp LISTS_ROOT, --lists-root LISTS_ROOT
                        root of sample lists. This must be downloaded from ekya repo.
  -dc, --use-data-cache
                        Use data caching for cityscapes. WARNING: Might
                        consume lot of disk space.
  -ir RESIZE_RES, --resize-res RESIZE_RES
                        Image size to use for cityscapes.
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of workers preprocessing the data.
  -ts TRAIN_SPLIT, --train-split TRAIN_SPLIT
                        Train validation split. This float is the fraction of
                        data used for training, rest goes to validation.
  -dtfs, --do-not-train-from-scratch
                        Do not train from scratch for every profiling task -
                        carry forward the previous model
  -hw HISTORY_WEIGHT, --history-weight HISTORY_WEIGHT
                        Weight to assign to historical samples when
                        retraining. Between 0-1. Cannot be zero. -1 if no
                        reweighting.
  -rp RESTORE_PATH, --restore-path RESTORE_PATH
                        Path to the pretrained models to use for init. Must be
                        downloaded from Ekya repo.
  -cp CHECKPOINT_PATH, --checkpoint-path CHECKPOINT_PATH
                        Path where to save the model
  -mn MODEL_NAME, --model-name MODEL_NAME
                        Model name. Can be resnetXX for now.
  -nc NUM_CLASSES, --num-classes NUM_CLASSES
                        Number of classes per task.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size.
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate.
  -mom MOMENTUM, --momentum MOMENTUM
                        Momentum.
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs per task.
  -nh NUM_HIDDEN, --num-hidden NUM_HIDDEN
                        Number of neurons in hidden layer.
  -dllo, --disable-last-layer-only
                        Adjust weights on all layers, instead of modifying
                        just last layer.
  -sched SCHEDULER, --scheduler SCHEDULER
                        Scheduler to use. Either of fair, noretrain, thief,
                        utilitysim.
  -usp UTILITYSIM_SCHEDULE_PATH, --utilitysim-schedule-path UTILITYSIM_SCHEDULE_PATH
                        Path to the schedule (period allocation) generated by
                        utilitysim.
  -usc UTILITYSIM_SCHEDULE_KEY, --utilitysim-schedule-key UTILITYSIM_SCHEDULE_KEY
                        The top level key in the schedule json. Usually of the
                        format {}_{}_{}_{}.format(period,res_count,scheduler,u
                        se_oracle)
  -mpd MICROPROFILE_DEVICE, --microprofile-device MICROPROFILE_DEVICE
                        Device to microprofile on - either of cuda, cpu or
                        auto
  -mprpt MICROPROFILE_RESOURCES_PER_TRIAL, --microprofile-resources-per-trial MICROPROFILE_RESOURCES_PER_TRIAL
                        Resources required per trial in microprofiling. Reduce
                        this to run multiple jobs in together while
                        microprofiling. Warning: may cause OOM error if too
                        many run together.
  -mpe MICROPROFILE_EPOCHS, --microprofile-epochs MICROPROFILE_EPOCHS
                        Epochs to run microprofiling for.
  -mpsr MICROPROFILE_SUBSAMPLE_RATE, --microprofile-subsample-rate MICROPROFILE_SUBSAMPLE_RATE
                        Subsampling rate while microprofiling.
  -mpep MICROPROFILE_PROFILING_EPOCHS, --microprofile-profiling-epochs MICROPROFILE_PROFILING_EPOCHS
                        Epochs to generate profiles for, per hyperparameter.
  -fswt FAIR_INFERENCE_WEIGHT, --fair-inference-weight FAIR_INFERENCE_WEIGHT
                        Weight to allocate for inference in the fair
                        scheduler.
  -nt NUM_TASKS, --num-tasks NUM_TASKS
                        Number of tasks to split each dataset into
  -stid START_TASK, --start-task START_TASK
                        Task id to start at.
  -ttid TERMINATION_TASK, --termination-task TERMINATION_TASK
                        Task id to end the Ekya loop at. -1 runs all tasks.
  -nsp NUM_SUBPROFILES, --num-subprofiles NUM_SUBPROFILES
                        Number of tasks to split each dataset into
  -op RESULTS_PATH, --results-path RESULTS_PATH
                        The josn file to write results to.
  -uhp HYPS_PATH, --hyps-path HYPS_PATH
                        hyp_map.json path which lists the hyperparameter_id to
                        hyperparameter mapping.
  -hpid HYPERPARAMETER_ID, --hyperparameter-id HYPERPARAMETER_ID
                        Hyperparameter id to use for retraining. From hyps-
                        path json.
  -pm, --profiling-mode
                        Run in profiling mode?
  -pp PROFILE_WRITE_PATH, --profile-write-path PROFILE_WRITE_PATH
                        Run in profiling mode?
  -ipp INFERENCE_PROFILE_PATH, --inference-profile-path INFERENCE_PROFILE_PATH
                        Path to the inference profiles csv
  -mir MAX_INFERENCE_RESOURCES, --max-inference-resources MAX_INFERENCE_RESOURCES
                        Maximum resources required for inference. Acts as a
                        ceiling for the inference scaling function.

```

## Golden model

Download resnext101 elastic model from [here](https://github.com/allenai/elastic)
into ```ekya/golden_model/```.

## Frequently Asked Questions

1. When installing ray with `pip install -e . --verbose` and encountering the 
   error `"[ray] [bazel] build failure, error --experimental_ui_deduplicate
   unrecognized"`. 

    Please checkout this
    [issue](https://github.com/ray-project/ray/issues/11237). If other versions
    of `bazel` are installed, please install `bazel-3.2.0` following instructions
    from
    [here](https://docs.bazel.build/versions/main/install-compile-source.html)
    and compile ray useing `bazel-3.2.0`.

