#!/bin/bash
#$ -N bert_training_fk_news_project
#$ -cwd
#$ -l h_rt=24:00:00
#$ -l mem=100G
#$ -pe smp 8
#$ -l gpu=1
#$ -o output.log
#$ -e error.log
#$ -m beas
#$ -M daniel.diaz@ucl.ac.uk

module load python/3.8
module load cuda/11.2
source /path/to/your/venv/bin/activate

python bert_training_THSK2023
