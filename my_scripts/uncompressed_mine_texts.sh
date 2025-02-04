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
#SBATCH --partition=devlab

export LASER="${HOME}/LASER"
K=4
TARGET_LNG="en"
SRC_LNG="de"

srun python3 source/mine_bitexts.py /private/home/marialomeli/LASER/tasks/bucc/embed/bucc2018.${SRC_LNG}-${TARGET_LNG}.train.txt.${SRC_LNG} \
 /private/home/marialomeli/LASER/tasks/bucc/embed/bucc2018.${SRC_LNG}-${TARGET_LNG}.train.txt.${TARGET_LNG} \
--src-lang ${SRC_LNG} --trg-lang ${TARGET_LNG} --src-embeddings  /private/home/marialomeli/LASER/tasks/bucc/embed/sonar_embeds/encf.bucc2018.${SRC_LNG}-${TARGET_LNG}.train.${SRC_LNG} \
--trg-embeddings /private/home/marialomeli/LASER/tasks/bucc/embed/sonar_embeds/encf.bucc2018.${SRC_LNG}-${TARGET_LNG}.train.${TARGET_LNG} --mode mine --retrieval max \
--margin ratio -k ${K} --output /private/home/marialomeli/LASER/tasks/bucc/embed --verbose --fp16 --unify  --gpu
