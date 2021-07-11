cd utils||exit
python processer.py --seed 7 --model_name "bert"
cd ..
python train.py --epoch 50 --mode "res14" --step 100 --loss "focal" --optimizer "adamw"