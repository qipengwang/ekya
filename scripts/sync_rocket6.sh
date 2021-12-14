#!/usr/bin/env bash
set -e
#rsync -ariz --progress --exclude '*.tar.gz' --exclude 'test' --exclude 'train' --exclude 'meta' '/home/romilb/research/msr/incremental_learning.pytorch/' rocket6:'/home/researcher/incremental_learning.pytorch/'
rsync -ariz --progress --exclude --delete '*.tar.gz' --exclude 'test' --exclude 'train' --exclude 'meta' --exclude '.git' '/ekya/ekya_root/' rocket6:'/home/researcher/ekya/'
#rsync -ariz --progress --exclude '*.tar.gz' '/home/romilb/datasets/cityscapes_raw/sample_lists/' rocket6:'/home/researcher/datasets/cityscapes/sample_lists/'
# rsync -ariz --progress --exclude '*.tar.gz' --exclude 'test' --exclude 'train' --exclude 'meta' '/home/romilb/research/msr/results/' rocket6:'/home/researcher/ekya_results/'
