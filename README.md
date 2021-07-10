# ABSA_E2E
占坑中，待完成。。。。。

```c
.
├── checkout
│   ├── data_processing_log.txt
│   └── state_dict
├── config
│   └── config.py
├── data
│   ├── Semeval2014
│   │   ├── processed // 预处理好的数据
│   │   └── raw //原XML文件
│   ├── Semeval2016
│   │   ├── processed
│   │   └── raw
│   └── stopwords.txt
├── models
│   └── BERT_BASE.py
├── README.md
├── test_demo.py
├── train.py
└── utils
    ├── data_utils.py  
    ├── metrics.py 
    └── processer.py  // 预处理函数

```

## 预处理

```python
cd utils
python processer.py --bpe  # process bert training file
```



运行`processer.py` 文件进行数据预处理 

## SemEval Official

[2016 task5](https://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools)

[2014 task4](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

