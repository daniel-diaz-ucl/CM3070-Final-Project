#!/bin/bash -l
#$ -N bert_training_fk_news_project
#$ -l h_rt=24:00:00
#$ -l mem=100G
#$ -pe smp 4
#$ -l gpu=1
#$ -o output.log
#$ -e error.log
#$ -m beas
#$ -M daniel.diaz@ucl.ac.uk
#$ -l tmpdir=32G
#$ -ac allow=L
#$ -wd /home/ucjtfdd/Scratch/fk_news_project

module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu

python bert_training_THSK2023.py
