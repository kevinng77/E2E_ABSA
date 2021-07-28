
# **Results**

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

**Seed** : seed for spliting the dataset

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

#### Focal Loss Tuning on bert

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

<u>The following model is trained and test on different dataset with split seed (6, 7, 8, 66, 77). Each model test on each dataset once. The final score is the mean of Macro F1 for 5 tests.</u>

|                 | **lap 14** | **lap 14** | **lap 14** | lap 14 | res 16 | res 16 | res 16 | res 16 | res 14 | res 14 | res 14 | res 14 |
| --------------- | ---------- | ---------- | ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| models          | AE         | PC         | CE         | BP     | AE     | PC     | CE     | BP     | AE     | PC     | CE     | BP     |
| glove-lstm      | 82.48      | 59.53      | 54.89      | 0.0    | 83.32  | 60.21  | 55.40  | 0.0    | 86.78  | 62.71  | 60.03  | 0.0    |
| glove- LSTM-CRF | 84.62      | 63.19      | 60.13      | 0.0    | 86.54  | 64.19  | 60.51  | 0.0    | 88.25  | 64.17  | 63.31  | 0.0    |

glove-san, glove-linear performs terrible in parameter tuning section, therefore do not conduct the testing for them.

|             | **lap 14** | **lap 14** | **lap 14** | lap 14 | res 16    | res 16    | res 16    | res 16 | res 14    | res 14    | res 14    | res 14 |
| ----------- | ---------- | ---------- | ---------- | ------ | --------- | --------- | --------- | ------ | --------- | --------- | --------- | ------ |
| models      | AE         | PC         | CE         | BP     | AE        | PC        | CE        | BP     | AE        | PC        | CE        | BP     |
| bert-linear | 87.60      | 70.14      | 64.80      | 0      | 85.30     | 67.11     | 62.34     | 0      | 89.49     | 72.04     | 68.13     | 0      |
| bert-lstm   | 87.07      | **71.31**  | 65.01      | 0      | 85.57     | **70.83** | **64.93** | 0      | **90.23** | 72.20     | 68.87     | 0      |
| bert-san    | 87.08      | 69.57      | 63.94      | 0      | 85.09     | 67.15     | 61.88     | 0      | 90.01     | **74.46** | **70.12** | 0      |
| bert-crf    | **87.80**  | 69.97      | **65.07**  | 0      | **85.73** | 69.12     | 64.20     | 0      | 89.97     | 72.82     | 68.72     | 0      |

Refered to the Breaking Prediction Score, that is how many times the model extract a complete word with inconsistent polarity. It indicate the bert is good enough to generate a prediction without broken, such that the CRF brings less benifit to it rather than to ELMO and GLOVE.

|                              | **lap 14** | **lap 14** | **lap 14** | lap 14 | res 16    | res 16    | res 16    | res 16 | res 14    | res 14    | res 14    | res 14 |
| ---------------------------- | ---------- | ---------- | ---------- | ------ | --------- | --------- | --------- | ------ | --------- | --------- | --------- | ------ |
| models                       | AE         | PC         | CE         | BP     | AE        | PC        | CE        | BP     | AE        | PC        | CE        | BP     |
| elmo-lstm                    | 86.00      | **68.44**  | 62.32      | 0.0    | 86.22     | 63.42     | 59.60     | 0.0    | 89.26     | 67.49     | 65.17     | 0.0    |
| elmo-san                     | 84.75      | 62.46      | 58.60      | 0.0    | 87.08     | **64.97** | **61.58** | 0.0    | 88.44     | **69.50** | 66.05     | 0.0    |
| elmo-crf                     | 84.62      | 63.19      | 60.13      | 0.0    | 86.54     | 64.19     | 60.51     | 0.0    | 88.25     | 64.17     | 63.31     | 0.0    |
| elmo-lstm-crf                | **87.56**  | 65.91      | **62.37**  | 0      | **87.11** | 61.91     | 60.17     | 0      | **89.76** | 67.05     | 65.06     | 0      |
| ELMO-lstm-crf (no fine-tune) | 85.73      | 67.11      | 61.59      | 0.0    | 86.74     | 57.81     | 56.85     | 0.0    | 89.63     | 68.82     | **66.49** | 0      |

#### R-drop and contrastive loss test:

<u>The following result is test on dataset with split seed 6. Each model was train and test on the dataset 5 times. The score is the mean of Macro F1 for 5 tests.</u> 

baseline model: bert-base-uncased + linear + focal loss (alpha 0.75,  gamma 2)

r-drop: r_drop alpha = 0.01 (The paper use 1 together with CE, considering focal loss is much less than CE)

contrastive loss: temperature = 0.05, alpha = 0.1, selected small alpha same reason as above.

|                  | **lap 14** | **lap 14** | **lap 14** | lap 14 | res 16 | res 16 | res 16 | res 16 | res 14 | res 14 | res 14 | res 14 |
| ---------------- | ---------- | ---------- | ---------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| loss type        | AE         | PC         | CE         | BP     | AE     | PC     | CE     | BP     | AE     | PC     | CE     | BP     |
| focal los        | 87.72      | 72.52      | 66.47      | 0      | 84.83  | 69.86  | 63.07  | 0      | 89     | 72.15  | 68.07  | 0      |
| rdrop            | 87.91      | 73.29      | 67.38      | 0      | 84.63  | 72.85  | 66.88  | 0      | 89.38  | 70.5   | 67.16  | 0      |
| contrastive loss | 87.49      | 70.59      | 65.07      | 0      | 85.35  | 71.38  | 65.18  | 0      | 89.15  | 71.43  | 67.75  | 0      |

**Reference:**

[[1\] ](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+T)[Tianyu](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+T)[ Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+T), [Xingcheng](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+X)[ Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+X), [Danqi](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+D)[ Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+D) 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. [arXiv:2104.08821](https://arxiv.org/abs/2104.08821) **[cs.CL]**

[2] [Xiaobo Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+X), [Lijun Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+L), [Juntao](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J)[ Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Yue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Qi Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+Q), [Tao Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+T), [Wei Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+W), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Tie-Yan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T) 2021. R-Drop: Regularized Dropout for Neural Networks [arXiv:2106.14448](https://arxiv.org/abs/2106.14448) **[cs.LG]**

