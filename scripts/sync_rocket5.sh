#!/usr/bin/env bash
rsync -ariz --exclude '*.tar.gz' --exclude 'test' --exclude 'train' --exclude 'meta' /home/romilb/research/msr/incremental_learning.pytorch/ rocket5:/home/mnr/incremental_learning.pytorch/
rsync -ariz --exclude '*.tar.gz' --exclude 'test' --exclude 'train' --exclude 'meta' /home/romilb/research/msr/ekya/ rocket5:/home/mnr/ekya/
