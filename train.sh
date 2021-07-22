#!/bin/bash
seed=6
model_name="bert"
cd utils||exit
python processer.py --seed ${seed} --model_name ${model_name}
cd ..
for downstream in "lstm" "san" "crf"
do
  for mode in "res14" "lap14" "res16"
  do
    python train.py --clip_large_grad --mode ${mode} --model_name ${model_name} --downstream ${downstream} --seed ${seed} --gamma 2 --alpha 0.75 --optimizer "adamw"
    python test.py --mode ${mode} --model_name ${model_name} --downstream ${downstream} --seed ${seed}
  done
done
# downstream="linear"
# for mode in "res14" "lap14" "res16"
# do
#   python train.py --clip_large_grad --mode ${mode} --model_name ${model_name} --downstream ${downstream} --seed ${seed} --gamma 2 --alpha 0.75 --optimizer "adamw"
#   python test.py --mode ${mode} --model_name ${model_name} --downstream ${downstream} --seed ${seed}
# done