import sys
import torch
from torch.utils.data import DataLoader
from transformers import BertModel
from config import config
from models.pretrain_model import PretrainModel
from datetime import datetime
import pathlib
abs_path = pathlib.Path(__file__).absolute().parent
sys.path.append(str(abs_path.joinpath("utils")))
from utils import result_helper
from utils.data_utils import E2EABSA_dataset, Tokenizer
from utils.metrics import F1
from utils.result_helper import init_logger
from utils.processer import process_string
from allennlp.modules.elmo import Elmo


def load_model():
    args = config.args
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(args=args)
    if args.model_name.startswith("bert"):
        bert = BertModel.from_pretrained(args.pretrained_bert_name)
        model = PretrainModel(pretrain_model=bert, args=args).to(args.device)
    else:
        elmo = Elmo(options_file=config.options_file,
                    weight_file=config.weight_file,
                    num_output_representations=1,
                    dropout=0,
                    requires_grad=True)
        model = PretrainModel(pretrain_model=elmo, args=args).to(args.device)

    model.load_state_dict(torch.load(args.state_dict_path))
    return model, tokenizer, args


def test(logger):
    model, tokenizer, args = load_model()
    model.eval()
    metrics = F1(num_classes=args.num_classes, downstream=args.downstream)
    torch.autograd.set_grad_enabled(False)
    test_dataloader = DataLoader(E2EABSA_dataset(file_path=args.file_path['test'],
                                                 tokenizer=tokenizer),
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 drop_last=False)

    confusion = torch.zeros([4, 4], device=args.device)
    aspect = torch.zeros([2, 2], device=args.device)
    broken = 0
    with torch.no_grad():
        for data in test_dataloader:
            inputs = data["text_ids"].to(args.device)
            target = data["pred_ids"].to(args.device)
            attention_mask = data["att_mask"].to(args.device)

            if not args.downstream.endswith("crf"):
                output = model(inputs, attention_mask=attention_mask)
                pred = torch.argmax(output, dim=-1).view(-1)
            else:
                _, logits = model(inputs, attention_mask=attention_mask, labels=target)
                output = model.crf.viterbi_tags(logits=logits, mask=attention_mask)
                # padding outputs
                output = [x + [tokenizer.target_pad_token_id] * (args.max_seq_len - len(x))
                          for x in output]
                pred = torch.tensor(output, dtype=torch.long, device=args.device).view(-1)

            d_aspect, d_confusion, d_broken = result_helper.gen_confusion_matrix(outputs=pred,
                                                                                 targets=target.view(-1))
            aspect = aspect + d_aspect.to(args.device)
            confusion = confusion + d_confusion.to(args.device)
            broken += d_broken

    f1_aspect, f1_polarity, f1_total = result_helper.gen_metrics(confusion)
    logger.info(f'{model.model_name}\t'
                f'{args.mode}\t{metrics.name}\t'
                f'aspect {f1_aspect * 100:.2f}%\t'
                f'polarity {f1_polarity * 100:.2f}%\t'
                f'total {f1_total * 100:.2f}%\t'
                f'#broken {broken}\t'
                f'seed {args.seed}\t'
                f'{datetime.now()}')


def demo():
    model, tokenizer, args = load_model()
    model.eval()
    torch.autograd.set_grad_enabled(False)
    print("input your sentence: ")
    with torch.no_grad():
        while 1:
            a = sys.stdin.readline().strip()
            if a == 'exit':
                break
            token_list = tokenizer.text_to_ids(process_string(a))
            len_seq = len(process_string(a).split())
            if args.model_name == "bert":
                attention_mask = torch.tensor(
                    [1 if x != 0 else 0 for x in token_list]).view(1, -1).to(args.device)
                inputs = torch.tensor(token_list).view(1, -1).to(args.device)
            else:
                inputs = token_list.unsqueeze(0).long().to(args.device)
                attention_mask = torch.tensor(
                    [1]*len_seq + [0] * (args.max_seq_len - len_seq),device=args.device
                ).view(1,-1)
            if not args.downstream.endswith("crf"):
                outputs = model(inputs, attention_mask=attention_mask)
                outputs = torch.masked_select(torch.argmax(outputs, dim=-1),
                                              attention_mask == 1).cpu().numpy()
            else:
                _, logits = model(inputs, attention_mask=attention_mask, labels=attention_mask)
                output = model.crf.viterbi_tags(logits=logits, mask=attention_mask)
                outputs = torch.tensor(output, dtype=torch.long, device=args.device).view(-1).cpu().numpy()
            pred = tokenizer.ids_to_tokens(outputs, is_target=True)
            print(pred)


if __name__ == "__main__":
    if config.args.demo:
        demo()
    else:
        logger = init_logger(logging_folder=config.working_path.joinpath('checkout'),
                             logging_file=config.working_path.joinpath("checkout/test_log.txt"))
        test(logger)
