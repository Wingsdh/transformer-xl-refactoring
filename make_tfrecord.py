# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    make_tfrecord.py
   Description :
   Author :       Wings DH
   Time：         2020/1/18 9:49 上午
-------------------------------------------------
   Change Activity:
                   2020/1/18: Create
-------------------------------------------------
"""
from enum import Enum
import os

import tensorflow as tf
from absl import flags

import common.default_config as cfg
from common.log import logger
from data_processing.standard_generator import DirCorpusGenerator, FileCorpusGenerator, MixCorpusGenerator
from data_processing.tfrecord_process import TFRecordMaker
from data_processing.tokenizer import CharTokenizer, SpaceTokenizer, CustomTokenizer
from data_processing.vocabulary import Vocabulary, SentencePieceVocabulary


class CorpusType(Enum):
    FILE = 'file'
    DIR = 'dir'
    WIKI2019 = 'wiki2019zh'


class TokenizerType(Enum):
    CHAR = 'char'
    SPACE = 'space'
    JIEBA = 'jieba'


def parse_path_and_type():
    paths_str = FLAGS.data_paths
    if not paths_str:
        raise ValueError('Must add a data_paths')
    paths = [p.strip() for p in paths_str.split(',')]

    type_corpus_gen_str = FLAGS.type_corpus_gens
    if not type_corpus_gen_str:
        raise ValueError('Must add a type_corpus_gens')
    type_corpus_gens = [t.strip() for t in type_corpus_gen_str.split(',')]

    if len(paths) != len(type_corpus_gens):
        raise ValueError(
            'Paths must same number with type_corpus_gens, {} <> {}'.format(len(paths), len(type_corpus_gens)))

    return [(p, t) for p, t in zip(paths, type_corpus_gens)]


def check_paths(path_types):
    # 语料路径必须存在
    for p, _ in path_types:
        if not os.path.exists(p):
            raise ValueError('Oh,Oh, Data dir <{}> not exist'.format(p))

    if not os.path.exists(FLAGS.tfrecord_d_path):
        logger.info('TFRecord dir <{}> not exist, create it'.format(FLAGS.tfrecord_d_path))
        os.makedirs(FLAGS.tfrecord_d_path)


def build_tokenizer():
    type_tokenizer = FLAGS.type_tokenizer
    if type_tokenizer == TokenizerType.CHAR.value:
        return CharTokenizer.new_instance()

    elif type_tokenizer == TokenizerType.SPACE.value:
        return SpaceTokenizer.new_instance()

    elif type_tokenizer == TokenizerType.JIEBA.value:
        import jieba
        return CustomTokenizer.new_instance(func=lambda text: jieba.lcut(text))

    else:
        return None
        # raise ValueError('Unknown tokenizer type: {}'.format(type_tokenizer))


def build_corpus_iter(path_types):
    its = []
    for data_path, type_corpus_gen in path_types:
        if type_corpus_gen == CorpusType.FILE.value:
            it = FileCorpusGenerator.new_instance(data_path)

        elif type_corpus_gen == CorpusType.DIR.value:
            it = DirCorpusGenerator.new_instance(data_path, recursive=True, check_func=lambda x: x.endswith('txt'))

        elif type_corpus_gen == CorpusType.WIKI2019.value:
            import json

            def is_non_cn_char_line(text, threshold=0.8):
                num_cn_char = len([c for c in text if not '\u4e00' <= c <= '\u9fa5'])
                if num_cn_char / len(text) > threshold:
                    return True
                else:
                    return False

            def split_line(text):
                json_string = json.loads(text)
                text = json_string.get('text', '')
                # 只取长度大于10的中文文本
                return [line for line in text.split('\n') if len(line) > 10 and not is_non_cn_char_line(line)]

            it = DirCorpusGenerator(
                data_path, recursive=True, split_func=split_line)

        else:
            raise ValueError('Unknown corpus iter type: {}'.format(type_corpus_gen))
        its.append(it)
    return MixCorpusGenerator.new_instance(its)


def build_vocabulary(tokenizer, corpus_iter=None):
    vocab_f_path = FLAGS.vocab_path
    vocab_type = FLAGS.vocab_type
    vocab = None
    if vocab_type == 'sentence_piece':
        vocab = SentencePieceVocabulary.new_instance(path=vocab_f_path)
    elif vocab_type == 'default':
        if os.path.exists(vocab_f_path):
            vocab = Vocabulary.new_from_save_file(vocab_f_path, tokenizer=tokenizer)
        else:
            vocab = Vocabulary.new_from_corpus(corpus_iter, tokenizer=tokenizer,
                                               max_n_tokens=FLAGS.max_n_token,
                                               min_freq=FLAGS.min_freq)
            Vocabulary.save_vocab(vocab, vocab_f_path)
    return vocab


def main(unused_argv):
    del unused_argv  # Unused

    path_types = parse_path_and_type()

    # 校验路径
    check_paths(path_types)

    # 语料迭代器
    corpus_iter = build_corpus_iter(path_types)

    # 分词器
    tokenizer = build_tokenizer()

    # 词库
    vocab = build_vocabulary(corpus_iter=corpus_iter, tokenizer=tokenizer)

    # 制作tfrecord文件
    tf_record_maker = TFRecordMaker(corpus_iter, vocab, dataset_name=FLAGS.dataset, add_start=True, add_end=True)
    tf_record_maker.convert_all_tfrecords(FLAGS.tfrecord_d_path, FLAGS.batch_size, FLAGS.tgt_len,
                                          record_filename=FLAGS.record_filename)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("dataset", default=cfg.DEFAULT_DATASET, help="Dataset name.")
    flags.DEFINE_string("data_paths", cfg.DEFAULT_DATA_ROOT, help="Paths of the data corpus, dir path or file path")
    flags.DEFINE_string("type_corpus_gens", cfg.DEFAULT_TYPE_CORPUS_GENERATOR, help="Type of corpus generator")
    flags.DEFINE_string("file_path", cfg.DEFAULT_DATA_FILE, help="File of the data corpus")
    flags.DEFINE_string("vocab_path", cfg.DEFAULT_VOCAB_FILE, help="Vocab path where save vocab")
    flags.DEFINE_string("vocab_type", cfg.DEFAULT_VOCAB_TYPE, help="Vocab type")
    flags.DEFINE_string("tfrecord_d_path", cfg.DEFAULT_TFRECORDS_D_PATH, help="Tfrecords path where save tfrecords")
    flags.DEFINE_string("record_filename", cfg.DEFAULT_RECORD_FILENAME, help="Record json filename")

    flags.DEFINE_string("type_tokenizer", cfg.DEFAULT_TYPE_TOKENIZER, help="Type of corpus generator")
    flags.DEFINE_integer("max_n_token", cfg.DEFAULT_MAX_N_TOKEN, help="Max num of tokens")
    flags.DEFINE_integer("min_freq", cfg.DEFAULT_MIN_FREQ, help="Min freq of token in corpus")

    flags.DEFINE_integer("batch_size", cfg.DEFAULT_BATCH_SIZE, help="train batch size each host")
    flags.DEFINE_integer("tgt_len", cfg.DEFAULT_TGT_LEN, help="number of tokens to predict")
    flags.DEFINE_integer("num_core_per_host", default=cfg.DEFAULT_NUM_CORE, help="Number of cores per host")
    tf.app.run(main)
