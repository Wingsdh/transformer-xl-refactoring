#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :   Wingsdh

@File    :   default_config.py

@Time    :   2020/1/18 11:32

@Desc    :

'''
from corpus_generator.standard_generator import FileCorpusGenerator

# 数据配置
from data_processing.tokenizer import SpaceTokenizer

DEFAULT_DATASET = 'unknown_data'
DEFAULT_DATA_ROOT = '../data/{}/'.format(DEFAULT_DATASET)
DEFAULT_DATA_FILE = '../data/{}/corpus.txt'.format(DEFAULT_DATASET)
DEFAULT_TFRECORDS_D_PATH = 'tfrecords/{}/'.format(DEFAULT_DATASET)
DEFAULT_VOCAB_FILE = '{}/vocab.txt'.format(DEFAULT_TFRECORDS_D_PATH)
DEFAULT_RECORD_FILENAME = 'record.json'
DEFAULT_TYPE_CORPUS_GENERATOR = FileCorpusGenerator.TYPE
DEFAULT_TYPE_TOKENIZER = SpaceTokenizer.TYPE

# 公共配置
DEFAULT_TGT_LEN = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_CORE = 1

# 词库配置
USE_VOCAB_FILE = True
VOCAB_SZ = 0

# 训练配置
DEFAULT_NUM_CORE_PER_HOST = 1
TRAIN_STEPS = 350000
MEM_LEN = 100  # Number of steps to cache

# 模型
N_LAYER = 16
D_MODEL = 512
D_EMBED = 512
N_HEAD = 10
D_HEAD = 41
D_INNER = 1024
DROPOUT = 0.3
