import argparse
import pathlib
working_path = pathlib.Path(__file__).absolute().parent.parent
parser = argparse.ArgumentParser()

# Preprocessing
parser.add_argument('--max_seq_len', type=int, default=128, help="default 128")
parser.add_argument('--pretrained_bert_name', type=str, default='bert-base-uncased',
                    help='used only when model_name is bert. i.e. bert-base-uncased')
parser.add_argument('--split_ratio', type=float, nargs='+',
                    default=[0.8, 0.1, 0.1], help='train_ratio dev_ratio test_ratio')

# model param
parser.add_argument("--dropout", type=float, default=0.1,help="default 0.1")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="default 1e-4")
parser.add_argument("--num_classes", type=int, default=10,
                    help="number of target labels. default 10 for O, [B-,I-,E-]*[pos, neg, neu]")
parser.add_argument("--batch_size", type=int, default=32, help="default 32")
parser.add_argument("--lr", type=float, default=5e-5, help="default 5e-5")
parser.add_argument("--epochs", type=int, default=100, help="default 100")
parser.add_argument("--step", type=int, default=100,
                    help="checkout for each _ number of training step, default 100")
parser.add_argument("--model_name",type=str,default='bert',help="bert or elmo")
parser.add_argument("--downstream",type=str,default="linear",
                    help="linear, crf, lstm, san or lstm-crf")

# downstream attention heads
parser.add_argument("--num_heads",type=int,default=12,help="Default 12. number of attention heads for additional SAN")
parser.add_argument("--num_layers",type=int,default=1,help="Default 1. number of LSTM layers")

#  training param
parser.add_argument("--optimizer",type=str, default="adamw")
parser.add_argument("--loss",type=str,default="focal",help="'CE' or 'focal")
parser.add_argument("--gamma",type=float,default=2.0,help="gamma for focal loss")
parser.add_argument("--alpha",type=float,default=0.75,help="alpha for focal loss")
parser.add_argument("--shuffle", action='store_true', default=False)
parser.add_argument("--load_model", action='store_true', default=False)
parser.add_argument("--state_dict_path", type=str, default="bert_res14_F1_59.03.pth")
parser.add_argument("--seed", type=int, default=7,help="default 7")
parser.add_argument("--metrics",type=str,default="f1", help="f1.")
parser.add_argument("--verbose", action='store_true', default=False)
parser.add_argument("--warmup_steps",type=int,default=500)
parser.add_argument("--max_steps",type=int,default=3000)
parser.add_argument("--max_grad_norm",type=float,default=2.0,help="limit max grad norm")
parser.add_argument("--clip_large_grad",default=False,action='store_true',
                    help="clip large gradient before update optimize")

# optimizer param
parser.add_argument("--adam_beta1",type=float,default=0.9)
parser.add_argument("--adam_beta2",type=float,default=0.999)
parser.add_argument("--adam_epsilon",type=float,default=1e-8)
parser.add_argument("--adam_amsgrad", default=False, action='store_true')

# path
parser.add_argument('--mode',type=str,default="debug",help="default debug, res14, res16 or lap14")

parser.add_argument('--finetune_elmo',default=False,action='store_true',help="ELMo.trainable = True if provided")

# test or demo
parser.add_argument("--demo",default=False,action='store_true')

args = parser.parse_args()


if args.load_model:
    args.state_dict_path = "checkout/state_dict/" + args.state_dict_path
else:
    args.state_dict_path = f'checkout/state_dict/{args.model_name}-{args.downstream}_' \
                       f'{args.mode}_seed{args.seed}.pth'

# path to elmo data
options_file = "data/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "data/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# raw data to be process
raw_data_path = {
    "res14": str(working_path.joinpath("data/Semeval2014/raw/Restaurants_Train_v2.xml")),
    "res16": str(working_path.joinpath("data/Semeval2016/raw/ABSA16_Restaurants_Train_SB1_v2.xml")),
    "lap14": str(working_path.joinpath("data/Semeval2014/raw/Laptop_Train_v2.xml"))
}

# path to save processed data
processed_data_path = {
    "res14":{
        "train": str(working_path.joinpath("data/Semeval2014/processed/Restaurants_Train_v2.csv")),
        "dev"  : str(working_path.joinpath("data/Semeval2014/processed/Restaurants_dev_v2.csv")),
        "test" : str(working_path.joinpath("data/Semeval2014/processed/Restaurants_test_v2.csv"))
    },
    "res16":{
        "train": str(working_path.joinpath("data/Semeval2016/processed/ABSA16_Restaurants_Train_SB1_v2.csv")),
        "dev"  : str(working_path.joinpath("data/Semeval2016/processed/ABSA16_Restaurants_dev_SB1_v2.csv")),
        "test" : str(working_path.joinpath("data/Semeval2016/processed/ABSA16_Restaurants_test_SB1_v2.csv"))
    },
    "lap14":{
        "train": str(working_path.joinpath("data/Semeval2014/processed/Laptop_Train_v2.csv")),
        "dev"  : str(working_path.joinpath("data/Semeval2014/processed/Laptop_dev_v2.csv")),
        "test" : str(working_path.joinpath("data/Semeval2014/processed/Laptop_test_v2.csv"))
    },
    'debug': {
        'train': str(working_path.joinpath('data/Semeval2014/processed/debug.csv')),
        'dev'  : str(working_path.joinpath('data/Semeval2014/processed/debug.csv')),
        'test' : str(working_path.joinpath('data/Semeval2014/processed/debug.csv'))
    },
}

args.file_path = processed_data_path[args.mode]