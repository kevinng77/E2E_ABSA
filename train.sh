#!/bin/bash
for seed in 6 7 66 77 666
do
  cd utils||exit
  python processer.py --seed ${seed} --model_name "bert" --working_path "../"
  cd ..
  for downstream in "linear" "lstm" "self_attention" "crf"
  do
    for mode in "res14" "lap14" "res16"
    do
      python train.py --mode ${mode} --loss "focal" --downstream ${downstream} --seed ${seed}
      python test.py --mode ${mode} --downstream ${downstream} --seed ${seed}
    done
  done
done
