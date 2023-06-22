#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --job-name=mine_texts
#SBATCH --output=/checkpoint/marialomeli/jobs/%A
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --partition=learnlab

export LASER="${HOME}/LASER"
CODE_SIZE=256
K=200
TARGET_LNG="en"
SRC_LNG="fr"
OUTPUT
srun python3 source/mine_bitexts.py /private/home/marialomeli/LASER/tasks/bucc/embed/bucc2018.${SRC_LNG}-${TARGET_LNG}.train.txt.${SRC_LNG} \
 /private/home/marialomeli/LASER/tasks/bucc/embed/bucc2018.${SRC_LNG}-${TARGET_LNG}.train.txt.${TARGET_LNG} \
--src-lang de --trg-lang en --src-embeddings  /private/home/marialomeli/LASER/tasks/bucc/embed/sonar_embeds/encf.bucc2018.${SRC_LNG}-${TARGET_LNG}.train.${SRC_LNG} \
--trg-embeddings /private/home/marialomeli/LASER/tasks/bucc/embed/sonar_embeds/encf.bucc2018.${SRC_LNG}-${TARGET_LNG}.train.${TARGET_LNG} --mode mine --retrieval max \
--margin ratio -k ${K} --code_size ${CODE_SIZE} --output /private/home/marialomeli/LASER/tasks/bucc/embed/sonar${CODE_SIZE}.k${K}.bucc2018.${SRC_LNG}-${TARGET_LNG}.train.candidates.tsv --verbose --fp16 --unify 
