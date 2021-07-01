import argparse
parser = argparse.ArgumentParser()

# Preprocessing
parser.add_argument('--max_seq_len', type=int, default=128)
parser.add_argument('--pretrained_bert_name', type=str, default='bert-base-uncased',
                    help='bert-base-uncased')
parser.add_argument('--split_ratio', type=float, nargs='+',
                    default=[0.8, 0.1, 0.1], help='[train_ratio, dev_ratio, test_ratio]')

# model param
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--l2reg", type=float, default=1e-5)
parser.add_argument("--num_classes", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--step", type=int, default=500,
                    help="checkout for each _ number of training step")
parser.add_argument("--model_name",type=str,default='bert')

#  training param
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--shuffle", action='store_true', default=False)
parser.add_argument("--load_model", action='store_true', default=False)
parser.add_argument("--state_dict_path", type=str, default="2021-06-16 08:40:51_val_acc_0.34.pth")
parser.add_argument("--seed", type=int, default=7)

# path
parser.add_argument('--working_path', type=str, default="/home/kevin/nut/1_project/absa/",)
parser.add_argument('--mode',type=str,default="res14",help="res14, res16, lap14 or debug")

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
    },
    'debug': {
        'train':working_path + 'data/Semeval2014/processed/debug.csv',
        'dev':working_path + 'data/Semeval2014/processed/debug.csv',
        'test':working_path + 'data/Semeval2014/processed/debug.csv'
    },
}