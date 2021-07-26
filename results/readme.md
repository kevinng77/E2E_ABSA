
# Results

default settings:

batch_size 32 

max_seq_len 128 

max_steps 3000 

schedular: [get_linear_schedule_with_warmup](https://huggingface.co/transformers/main_classes/optimizer_schedules.html)

#### Columns:

**Dev f1**: Macro f1 for 9 classes during training. (B-pos, I-pos, E-pos, B-neg, I-neg, E-neg, B-neu, I-neu, E-neu)

**Test f1 (CE)**: Macro f1 for 4 classes during testing. (Not aspect, aspect-pos, aspect-neg, aspect-neu). 

**AE (Aspect Extract) F1**: Macro f1 for 2 classes during testing. (Not aspect term, aspect term)

**PC (Polarity Classify) F1**: Macro f1 for 3 classes during testing. (aspect-pos, aspect-neg, aspect-neu)

**BP (Broken Prediction)**: Number of predictions with inconsistent polarity <u>for target aspect term</u>. i.e. B-neg, I-pos, E-pos

*(aspect-pos means the word is aspect term and the polarity is positive)*

#### Cross Entropy Loss tuning

Weight:

1: `[0.1, 0.8, 1.0, 1.0, 1.2, 1.2, 1.2, 1.0, 1.0, 1.0] `

2: `[0.05, 0.8, 1.0, 1.0, 1.2, 1.2, 1.2, 1.0, 1.0, 1.0] `

3: `[0.07, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.0] `

| weight | optimizer      | lr   | Dev f1    |
| ------ | -------------- | ---- | --------- |
| 1      | Adam + amsgrad | 2e-5 | 57.21     |
| 1      | adam           | 2e-5 | 57.91     |
| 1      | adam           | 5e-5 | **61.86** |
| 1      | adamw          | 2e-5 | 56.88     |
| 1      | adamw          | 5e-5 | 59.85     |
| 2      | adamw          | 2e-5 | 56.16     |
| 3      | adam           | 2e-5 | 56.86     |
| 3      | adamw          | 2e-5 | 57.68     |
| 3      | adam           | 5e-5 | 58.65     |

Cross Entropy weight is choosen based on 

#### Focal Loss Tuning

Step-to-99: Step to reach 99% F1 score on train set.

| gamma | alpha | lr   | Dev f1    | Test f1   | Aspect Extract | Polarity Classify | Step-to 99 | seed | optimizer |
| ----- | ----- | ---- | --------- | --------- | -------------- | ----------------- | ---------- | ---- | --------- |
| 2     | 0.25  | 5e-5 | 59.78     | 68.02     | 89.07          | 71.94             | 1300       | 6    | adamw     |
| 2     | 0.25  | 2e-5 | 56.78     | 68.81     | 88.34          | 73.95             | 2800       | 6    | adamw     |
| 2     | 0.75  | 7e-5 | 59.17     | 68.78     | 89.93          | 71.80             | 1200       | 6    | adamw     |
| 2     | 0.75  | 5e-5 | **61.11** | 69.56     | 89.55          | 73.55             | 1300       | 6    | adamw     |
| 2     | 0.75  | 3e-5 | 58.49     | **70.08** | 89.62          | 74.57             | 1900       | 6    | adamw     |
| 2     | 0.75  | 2e-5 | 57.75     | 69.56     | 88.65          | 74.90             | 2300       | 6    | adamw     |
| 2     | 1     | 5e-5 | 59.92     | 69.69     | **89.99**      | 74.06             | 1500       | 6    | adamw     |
| 2     | 1     | 2e-5 | 56.72     | 68.98     | 88.54          | 74.18             | 2800       | 6    | adamw     |
| 3     | 2     | 5e-5 | 59.41     | 67.77     | 89.16          | 71.77             | 1600       | 6    | adamw     |
| 3     | 0.75  | 5e-5 | 58.58     | 68.20     | 88.72          | 73.27             | 1800       | 6    | adamw     |
| 3     | 5     | 5e-5 | 58.55     | 68.67     | 88.79          | 73.64             | 1700       | 6    | adamw     |
| 3     | 0.75  | 2e-5 | 56.99     | **70.23** | 88.74          | 75.59             | 2800       | 6    | adamw     |
| 2     | 0.75  | 5e-5 | 64.29     | 74.33     | 91.14          | 78.87             | 1400       | 7    | adamw     |
| 2     | 0.25  | 5e-5 | 63.50     | 72.41     | 91.74          | 75.54             | 1400       | 7    | adamw     |
| 2     | 1     | 5e-5 | 62.83     | 72.49     | 91.35          | 75.78             | 1400       | 7    | adamw     |
| 3     | 0.75  | 5e-5 | 63.53     | 72.86     | 91.12          | 76.97             | 1400       | 7    | adamw     |

#### Optimizer

| gamma | alpha | lr   | Dev f1    | Test f1   | Aspect Extract | Polarity Classify | seed | optimizer |
| ----- | ----- | ---- | --------- | --------- | -------------- | ----------------- | ---- | --------- |
| 2     | 0.75  | 5e-5 | 59.57     | 68.43     | 88.87          | 73.30             | 6    | adafactor |
| 2     | 0.75  | 5e-5 | **61.11** | **69.56** | **89.55**      | **73.55**         | 6    | **adamw** |
| 2     | 0.75  | 5e-5 | 59.05     | 67.81     | 88.31          | 73.71             | 6    | adam      |



#### Warm up Tuning

testing on bert linear model with adamw optimizer

| gamma | alpha | lr   | Dev-f1 | Test-f1 | Test- aspect | Test-polarity | seed | warmup |
| ----- | ----- | ---- | ------ | ------- | ------------ | ------------- | ---- | ------ |
| 2     | 0.25  | 2e-5 | 56.78  | 68.81   | 88.34        | 73.95         | 6    | 500    |
| 2     | 0.25  | 2e-5 | 59.26  | 68.12   | 88.60        | 73.39         | 6    | 300    |
| 2     | 0.75  | 2e-5 | 61.36  | 69.79   | 90.89        | 71.92         | 7    | 300    |
| 2     | 0.75  | 2e-5 | 62.36  | 69.81   | 90.04        | 73.75         | 7    | 0      |
| 2     | 0.75  | 2e-5 | 62.66  | 68.59   | 90.92        | 70.36         | 7    | 500    |

Warm up steps should not affect the final score significantly. Therefore it was set to 500 as default.

#### Model results

|                              | **lap 14** | **lap 14** | **lap 14** | lap 14 | res 16 | res 16 | res 16 | res 16 | res 14 | res 14 | res 14 | res 14 |
| ---------------------------- | ---------- | ---------- | ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| models                       | AE         | PC         | CE         | BP     | AE     | PC     | CE     | BP     | AE     | PC     | CE     | BP     |
| GLOVE-tagging                |            |            |            |        |        |        |        |        |        |        |        |        |
| glove-                       |            |            |            |        |        |        |        |        |        |        |        |        |
| Glove-                       |            |            |            |        |        |        |        |        |        |        |        |        |
| bert-linear                  | 87.60      | 70.14      | 64.80      | 0      | 85.30  | 67.11  | 62.34  | 0      | 89.49  | 72.04  | 68.13  | 0      |
| bert-lstm                    | 87.07      | 71.31      | 65.01      | 0      | 85.57  | 70.83  | 64.93  | 0      | 90.23  | 72.20  | 68.87  | 0      |
| bert-san                     | 87.08      | 69.57      | 63.94      | 0      | 85.09  | 67.15  | 61.88  | 0      | 90.01  | 74.46  | 70.12  | 0      |
| bert-crf                     | 87.80      | 69.97      | 65.07      | 0      | 85.73  | 69.12  | 64.20  | 0      | 89.97  | 72.82  | 68.72  | 0      |
|                              |            |            |            |        |        |        |        |        |        |        |        |        |
| ELMO -                       |            |            |            |        |        |        |        |        |        |        |        |        |
| ELMO -                       |            |            |            |        |        |        |        |        |        |        |        |        |
| ELMO -                       |            |            |            |        |        |        |        |        |        |        |        |        |
| ELMO-lstm-crf (no fine-tune) | 85.73      | 67.11      | 61.59      | 0.0    | 86.74  | 57.81  | 56.85  | 0.0    | 89.63  | 68.82  | 66.49  | 0.5    |

