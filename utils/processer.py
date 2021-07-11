import logging
import os
import random
import re
import sys
import xml.dom.minidom
from xml.dom.minidom import parse

sys.path.append('..')
from config import config
from data_utils import Tokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not os.path.exists(config.working_path + 'checkout'):
    os.mkdir(config.working_path + 'checkout')
handler = logging.FileHandler(config.working_path + "checkout/data_processing_log.txt")
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


def process_string(s, remove_stop=False, args=config.args):
    s = re.sub(r'^"+', "", s.lower())  # remove beginning "
    s = re.sub(r'$"+', "", s)  # remove ending "
    # s = re.sub(r"<br />", r" ", s.lower())  # remove <br />
    # s = re.sub(r"\*+", "", s)
    s = re.sub(r'""', r'"', s)
    s = re.sub(r" he's", r" he is", s)
    s = re.sub(r" she's", r" she is", s)
    s = re.sub(r"(?<=\w)'d", r" would", s)
    s = re.sub(r"(?<=\w)'ll", r" will", s)
    s = re.sub(r"(?<=\w)'m", r" am", s)
    s = re.sub(r"(?<=\w)'re", r" are", s)
    s = re.sub(r"n't", r" not", s)
    s = re.sub(r"(?<=\w)'ve", " have", s)
    s = re.sub(r"([^\w ])", r" \1 ", s)  # separate '
    s = re.sub(r"(^[^\w]+)", r" \1 ", s)  # separate begging '
    s = re.sub(r"('$)", r" \1", s)  # separate ending '
    s = re.sub(r" +", r" ", s)
    s = s.split()
    if remove_stop:
        with open(args.working_path + "data/stopwords.txt", "r") as fp:
            stops = fp.readlines()
        stops = [x.strip() for x in stops]
        s = [w for w in s if not w in stops]
    return " ".join(s)


def gen_data(xml_file, output_train, output_dev, output_test,
             train_tokenizer=None, args=config.args):
    """
    output:
    """
    logger.info(f">>> processing {xml_file} .")
    DOMTree = xml.dom.minidom.parse(xml_file)
    collection = DOMTree.documentElement
    sentences = collection.getElementsByTagName("sentence")
    logger.info(f"> # sentence: {len(sentences)}")
    polarity_count = {'positive': 0, 'negative': 0, 'neutral': 0, 'conflict': 0}
    num_train, num_dev, num_test = 0, 0, 0
    if 'Semeval2014' in xml_file:
        parse14 = True
    else:
        parse14 = False
    with open(output_train, 'w')as fp_train, \
            open(output_dev, 'w')as fp_dev, \
            open(output_test, 'w')as fp_test:
        for s in sentences:
            text = s.getElementsByTagName('text')[0].childNodes[0].data.lower()
            if train_tokenizer:
                text = train_tokenizer.text_to_tokens(process_string(text))
            else:
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
                    if train_tokenizer:
                        term = train_tokenizer.text_to_tokens(process_string(term))
                    else:
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
    train_tokenizer = None
    if args.model_name == "bert":
        if max_seq_len and pretrained_bert_name:
            train_tokenizer = Tokenizer(max_seq_len, pretrained_bert_name)
    for file_type in ['res14', 'res16', 'lap14']:
        gen_data(xml_file=config.raw_data_path[file_type],
                 output_train=config.processed_data_path[file_type]['train'],
                 output_dev=config.processed_data_path[file_type]['dev'],
                 output_test=config.processed_data_path[file_type]['test'],
                 train_tokenizer=train_tokenizer,
                 args=args
                 )


if __name__ == "__main__":
    args = config.args
    assert len(args.split_ratio) == 3, \
        "split ratio for train, dev, test are all require."
    random.seed(args.seed)
    main(args.max_seq_len, args.pretrained_bert_name, args)
