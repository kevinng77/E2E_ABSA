# ABSA E2E
```c
.
├── checkout
│   ├── data_processing_log.txt
│   ├── state_dict  //saved model
│   ├── test_log.txt
│   └── training_log.txt
├── config
│   └── config.py
├── data
│   ├── elmo  //elmo pretrained models
│   │   ├── elmo_2x4096_512_2048cnn_2xhighway_options.json
│   │   └── elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
│   ├── glove  //glove pretrained embeddings
│   ├── Semeval2014
│   ├── processed //processed files
│   │   ├── Restaurants_dev_v2.csv
│   │   ├── Restaurants_test_v2.csv
│   │   └── Restaurants_Train_v2.csv
│   └── raw  //raw SemEval xml data file
│   │   ├── Laptops_Train.xml
│   │   ├── Laptop_Train_v2.xml
│   │   ├── Restaurants_Train_v2.xml
│   │   └── Restaurants_Train.xml
│   ├── Semeval2016
│   │   ├── processed
│   │   └── raw
│   └── stopwords.txt
├── models
│   ├── downstream.py  //Linear, LSTM, Self-Attention, CRF
│   └── pretrain_model.py  
├── README.md
├── requirements.txt
├── results    
│   ├── total_train_log.csv  // 194 training records
│   └── readme.md  // more results
├── test.py
├── train.py
├── train.sh
└── utils
    ├── data_utils.py
    ├── metrics.py
    ├── processer.py
    └── result_helper.py
```

## Experiment

**CE (Co-Extract) F1**: Macro f1 for 4 classes during testing. (Not aspect, aspect-pos, aspect-neg, aspect-neu). 

**AE (Aspect Extract) F1**: Macro f1 for 2 classes during testing. (Not aspect term, aspect term)

**PC (Polarity Classify) F1**: Macro f1 for 3 classes during testing. (aspect-pos, aspect-neg, aspect-neu)

**BP (Broken Prediction)**: Number of predictions with inconsistent polarity <u>for target aspect term</u>. i.e. B-neg, I-pos, E-pos

**Result for bert**

<u>The following model is trained and test on different dataset with split seed (6, 7, 8, 66, 77). Each model test on each dataset once. The final score is the mean of Macro F1 for 5 tests.</u>

|             | **lap 14** | **lap 14** | **lap 14** | res 16 | res 16    | res 16    | res 14    | res 14    | res 14    |
| ----------- | ---------- | ---------- | ---------- | ------ | --------- | --------- | --------- | --------- | --------- |
| **models**  | **AE**     | **PC**     | **CE**     | **AE** | **PC**    | **CE**    | **AE**    | **PC**    | **CE**    |
| bert-linear | 87.60      | 70.14      | 64.80      | 85.30  | 67.11     | 62.34     | 89.49     | 72.04     | 68.13     |
| bert-lstm   | 87.07      | **71.31**  | 65.01      | 85.57  | **70.83** | **64.93** | **90.23** | 72.20     | 68.87     |
| bert-san    | 87.08      | 69.57      | 63.94      | 85.09  | 67.15     | 61.88     | 90.01     | **74.46** | **70.12** |
| bert-crf    | 87.80      | 69.97      | **65.07**  | 85.73  | 69.12     | 64.20     | 89.97     | 72.82     | 68.72     |

More results or detail please referred to [result folder](results/)

## To Run

#### Step 1: Process raw data

```python
cd utils
python processer.py --model_name "bert" --seed 6 --max_seq_len 128
```

`--model_name` : "bert", "elmo" or "glove"

 `--seed`: selected random seed.

`--split_ratio 0.8 0.1 0.1` : split ratio for train, dev, test set

#### Step 2: train Model (cd E2E_ABSA folder)

```shell
python train.py --mode "res14" --downstream "san" --model_name "bert" --seed 6
```

`--mode` : res14 , res16 or lap14. The SemEval task to train on.

`--downstream` : linear, lstm, crf, lstm-crf or san. The downstream model.

`--model_name` :  "bert", "elmo" or "glove", same asstep 1.

`--seed` : seed for record training log, same as step 1.

some other default settings:

```shell
--lr 5e-5 --batch_size 32 --loss "focal" --gamma 2 --alpha 0.75 --max_seq_len 128 --optimizer "adamw" --warmup_steps 300 --max_steps 3000
```

training log path: `./checkout/training_log.txt`

#### Step 3: Generate test result

 ```shell
 python test.py --mode "res14" --downstream "san" --model_name "bert" --seed 6
 ```

testing log path: `./checkout/test_log.txt`

#### 关于本仓库的 colba demo（中文）

[colab 链接](https://colab.research.google.com/drive/1X5CZ1LY5d-_oo8RewrZ4st-DC5F1fXCe?usp=sharing)

## Reference

SemEval official

[2016 task5](https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)

[2014 task4](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

Pretrained ELMo File

[weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)

[options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)

