#!/bin/bash -l
#$ -N roberta_large_training_fk_news_project
#$ -l h_rt=24:00:00
#$ -l mem=64G
#$ -l tmpfs=15G
#$ -pe smp 2
#$ -l gpu=2
#$ -o output.log
#$ -e error.log
#$ -m beas
#$ -M daniel.diaz@ucl.ac.uk
#$ -ac allow=LEF
#$ -wd /home/ucjtfdd/Scratch/fk_news_project/roberta_large

cd /home/ucjtfdd/Scratch/fk_news_project/roberta_large

module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu

export PYTHONPATH=/home/ucjtfdd/.python3local/lib/python3.9/site-packages:$PYTHONPATH

python3 roberta_large_training_THSK2023.py

tar zcvf $HOME/Scratch/fk_news_project/roberta_large/files_from_job_$JOB_ID.tar.gz $TMPDIR

