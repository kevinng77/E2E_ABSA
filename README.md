# ABSA E2E
```c
.
├── checkout
│   ├── data_processing_log.txt
│   ├── state_dict
│   ├── test_log.txt
│   └── training_log.txt
├── config
│   └── config.py
├── data
│   ├── elmo  //elmo pretrained models
│   │   ├── elmo_2x4096_512_2048cnn_2xhighway_options.json
│   │   └── elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
│   ├── Semeval2014
│   │   ├── processed //processed files
│   │   │   ├── Restaurants_dev_v2.csv
│   │   │   ├── Restaurants_test_v2.csv
│   │   │   └── Restaurants_Train_v2.csv
│   │   └── raw  //raw SemEval xml data file
│   │       ├── Laptops_Train.xml
│   │       ├── Laptop_Train_v2.xml
│   │       ├── Restaurants_Train_v2.xml
│   │       └── Restaurants_Train.xml
│   ├── Semeval2016
│   │   ├── processed
│   │   └── raw
│   ├── stopwords.txt
├── models
│   ├── downstream.py  //Linear, LSTM, Self-Attention, CRF
│   └── pretrain_model.py  //BERT, ELMO
├── README.md
├── requirements.txt
├── test.py
├── train.py
├── train.sh
└── utils
    ├── checkout
    ├── data_utils.py
    ├── metrics.py
    ├── processer.py
    └── result_helper.py
```

## Experiment

### result

更新中

|      |      |      |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |
|      |      |      |

## To Run

#### Step 1: Process raw data

```python
cd utils
python processer.py --model_name "bert"  --working_path "../" --seed 6 --max_seq_len 128# process bert training file
```

`--model_name` : "bert" or "elmo"

`--working_path`: fixed to `"../"` or the local path to this repo.

 `--seed`: selected random seed.

`--split_ratio 0.8 0.1 0.1` : split ratio for train, dev, test set

#### Step 2: train Model (cd E2E_ABSA folder)

```shell
python train.py --mode "res14" --downstream "san" --model_name "bert" --seed 6
```

`--mode` : res14 , res16 or lap14. The SemEval task to train on.

`--downstream` : linear, lstm, crf or san. The downstream model.

`--model_name` : pretrained model name, keep consistent with step 1.

`--seed` : seed for record training log, same as step 1.

some other default settings:

```shell
--lr 5e-5 --batch_size 32 --loss "focal" --gamma 2 --alpha 0.75 --max_seq_len 128 --optimizer "adamw" --warmup_steps 300 --max_steps 3000
```

training log path: `./checkout/training_log.txt`

#### Step 3: Generate test result

 ```shell
 python test.py --mode "res14" --downstream "san" --model_name "bert"
 ```

testing log path: `./checkout/test_log.txt`

## Reference

SemEval official

[2016 task5](https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)

[2014 task4](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

Pretrained ELMo File

[weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)

[options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)

