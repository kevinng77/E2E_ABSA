import logging
import os
import sys


def bert_result_helper(sentence_list):
    pass


def init_logger(logging_folder, logging_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not os.path.exists(logging_folder):
        os.mkdir(logging_folder)
    handler = logging.FileHandler(logging_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(handler)
    return logger
