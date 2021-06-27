import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--pretrained_bert_name', type=str, default='bert-base-uncased',
                    help='bert-base-uncased')
parser.add_argument('--split_ratio', type=float, nargs='+',
                    default=[0.8, 0.1, 0.1], help='[train_ratio, dev_ratio, test_ratio]')

# path
parser.add_argument('--working_path', type=str, default="/home/kevin/nut/1_project/absa/",)
args = parser.parse_args()




working_path = args.working_path
raw_data_path = {
    "res14":working_path + "data/Semeval2014/raw/Restaurants_Train_v2.xml"
    ,
    "res16": working_path + "data/Semeval2016/raw/ABSA16_Restaurants_Train_SB1_v2.xml",
    "lap14": working_path + "data/Semeval2014/raw/Laptop_Train_v2.xml"
}

processed_data_path = {
    "res14":{
        "train":working_path+"data/Semeval2014/processed/Restaurants_Train_v2.csv",
        "dev":working_path+"data/Semeval2014/processed/Restaurants_dev_v2.csv",
        "test":working_path+"data/Semeval2014/processed/Restaurants_test_v2.csv"
    },
    "res16":{
        "train": working_path+"data/Semeval2016/processed/ABSA16_Restaurants_Train_SB1_v2.csv",
        "dev": working_path+"data/Semeval2016/processed/ABSA16_Restaurants_dev_SB1_v2.csv",
        "test": working_path+"data/Semeval2016/processed/ABSA16_Restaurants_test_SB1_v2.csv"
    },
    "lap14":{
        "train": working_path + "data/Semeval2014/processed/Laptop_Train_v2.csv",
        "dev":working_path +  "data/Semeval2014/processed/Laptop_dev_v2.csv",
        "test": working_path + "data/Semeval2014/processed/Laptop_test_v2.csv"
    }
}