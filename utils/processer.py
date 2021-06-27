from xml.dom.minidom import parse
import xml.dom.minidom
import re
import numpy as np
import os
import random
import logging
import sys
from config import config
# from transformers import BertTokenizer
# from data_utils import Tokenizer


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not os.path.exists('../checkout'):
    os.mkdir('../checkout')
handler = logging.FileHandler("../checkout/data_processing_log.txt")
handler.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(handler)


def match_string(origin, sub):
    for i in range(len(origin)):
        if origin[i] == sub[0]:
            for j in range(1, len(sub)):
                if sub[j] != origin[i + j] and sub[j] != origin[i + j][:-1]:
                    break
            else:
                return i, [origin[x] for x in range(i, i + len(sub))]


def process_string(s, remove_stop=False):
    s = re.sub(r'^"+', "", s)  # remove beginning "
    s = re.sub(r'$"+', "", s)  # remove ending "
    # s = re.sub(r"<br />", r" ", s.lower())  # remove <br />
    # s = re.sub(r"\*+", "", s)
    s = re.sub(r'""', r'"', s)
    s = re.sub(r"([^\w ])", r" \1 ", s)  # separate '
    s = re.sub(r"(^[^\w]+)", r" \1 ", s)  # separate begging '
    s = re.sub(r"('$)", r" \1", s)  # separate ending '
    s = re.sub(r"(?<=\w)'s", r" is", s)
    s = re.sub(r"(?<=\w)'d", r" would", s)
    s = re.sub(r"(?<=\w)'ll", r" will", s)
    s = re.sub(r"(?<=\w)'m", r" am", s)
    s = re.sub(r"(?<=\w)'re", r" are", s)
    s = re.sub(r"n't", r" not", s)
    s = re.sub(r"(?<=\w)'ve", " have", s)
    s = re.sub(r" +", r" ", s)
    s = s.split()
    if remove_stop:
        with open("data/stopwords.txt","r") as fp:
            stops = fp.readlines()
        stops = [x.strip() for x in stops]
        s = [w for w in s if not w in stops]
    return " ".join(s)


def gen_data(xml_file, output_train, output_dev, output_test,
             train_tokenizer=None, args=config.args):
    logger.info(f">>> processing {xml_file} .")
    DOMTree = xml.dom.minidom.parse(xml_file)
    collection = DOMTree.documentElement
    sentences = collection.getElementsByTagName("sentence")
    logger.info(f"> # sentence: {len(sentences)}")
    polarity_count = {'positive': 0, 'negative': 0, 'neutral': 0, 'conflict': 0}
    num_train, num_dev, num_test = 0,0,0
    if 'Semeval2014' in xml_file:
        parse14 = True
    else:
        parse14 = False
    with open(output_train, 'w')as fp_train, \
            open(output_dev, 'w')as fp_dev, \
            open(output_test, 'w')as fp_test:
        for s in sentences:
            text = s.getElementsByTagName('text')[0].childNodes[0].data.lower()
            # text = train_tokenizer.text_to_tokens(text)
            text = process_string(text).split()
            if len(text) > 0:
                tags = ['O' for _ in range(len(text))]
                if parse14:
                    terms = s.getElementsByTagName('aspectTerm')
                else:
                    terms = s.getElementsByTagName('Opinion')

                for t in terms:
                    if parse14:
                        term = t.getAttribute('term').lower()
                    else:
                        term = t.getAttribute('target').lower()
                    # term = train_tokenizer.text_to_tokens(term)
                    term = process_string(term).split()
                    polarity = t.getAttribute('polarity')
                    try:
                        polarity_count[polarity] += 1
                    except KeyError:
                        polarity_count[polarity] = 1
                    if term[0] != 'null' and polarity != 'conflict':
                        pos, term = match_string(text, term)
                        tags[pos] = "B-" + polarity[:3]
                        if len(term) > 1:
                            tags[pos + len(term) - 1] = "E-" + polarity[:3]
                            if len(term) > 2:
                                for i in range(1, len(term) - 1):
                                    tags[pos + i] = "I-" + polarity[:3]
                tag_string = " ".join(tags)
                rnd = random.random()
                if rnd < args.split_ratio[0]:
                    fp_train.write(" ".join(text) + '\n')
                    fp_train.write(tag_string + '\n')
                    num_train += 1
                elif rnd < args.split_ratio[0] + args.split_ratio[1]:
                    fp_dev.write(" ".join(text) + '\n')
                    fp_dev.write(tag_string + '\n')
                    num_dev += 1
                else:
                    fp_test.write(" ".join(text) + '\n')
                    fp_test.write(tag_string + '\n')
                    num_test += 1

    logger.info(f"> # aspect {sum(polarity_count.values())}")
    logger.info(f"> {polarity_count}")
    logger.info(f"> # train {num_train} # dev {num_dev} # test {num_test}")


def main(max_seq_len, pretrained_bert_name, args):

    # 14 res
    for file_type in ['res14','res16','lap14']:
        gen_data(xml_file= config.raw_data_path[file_type],
                 output_train=config.processed_data_path[file_type]['train'],
                 output_dev=config.processed_data_path[file_type]['dev'],
                 output_test=config.processed_data_path[file_type]['test']
                 )




    # work_path_14 = "../data/Semeval2014/"
    # input_file_14 = [x for x in os.listdir(work_path_14 + 'raw') if "Train" in x]
    #
    # train_file_14 = [work_path_14 + "processed/" + x[:-4] + ".csv" for x in input_file_14]
    # dev_file_14 = [re.sub("Train", "dev", x) for x in train_file_14]
    # test_file_14 = [re.sub("Train", "test", x) for x in train_file_14]
    #
    # train_tokenizer = Tokenizer(max_seq_len, pretrained_bert_name)
    # for i in range(len(input_file_14)):
    #     gen_data(work_path_14 + "raw/" + input_file_14[i], train_file_14[i], dev_file_14[i],
    #              test_file_14[i], train_tokenizer, args)
    #
    # work_path_16 = "../data/Semeval2016/"
    # input_file_16 = [x for x in os.listdir(work_path_16 + 'raw') if "Train" in x]
    #
    # train_file_16 = [work_path_16 + "processed/" + x[:-4] + ".csv" for x in input_file_16]
    #
    # dev_file_16 = [re.sub("Train", "dev", x) for x in train_file_16]
    # test_file_16 = [re.sub("Train", "test", x) for x in train_file_16]
    #
    # for i in range(len(input_file_16)):
    #     gen_data(work_path_16 + "raw/" + input_file_16[i], train_file_16[i],
    #              dev_file_16[i],test_file_16[i],train_tokenizer,args)


if __name__ == "__main__":
    args = config.args
    assert len(args.split_ratio) == 3, \
        "split ratio for train, dev, test are all require."
    random.seed(7)
    main(args.max_seq_len, args.pretrained_bert_name, args)
