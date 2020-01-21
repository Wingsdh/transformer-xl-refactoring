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
import os

import tensorflow as tf
from absl import flags

import common.default_config as cfg
from corpus_generator.standard_generator import DirCorpusGenerator, FileCorpusGenerator
from corpus_generator.wiki2019zh_generator import Wiki2019zhGenerator
from data_processing.tfrecord_process import TFRecordMaker
from data_processing.tokenizer import CharTokenizer, SpaceTokenizer, CustomTokenizer
from data_processing.vocabulary import Vocabulary


def build_tokenizer():
    if FLAGS.type_tokenizer == CharTokenizer.TYPE:
        return CharTokenizer.new_instance()

    elif FLAGS.type_tokenizer == SpaceTokenizer.TYPE:
        return SpaceTokenizer.new_instance()

    elif FLAGS.type_tokenizer == 'jieba':
        import jieba
        return CustomTokenizer.new_instance(func=lambda text: jieba.lcut(text))

    else:
        raise ValueError('Unknown tokenizer type: {}'.format(FLAGS.type_corpus_gen))


def build_corpus_iter():
    type_corpus_gen = FLAGS.type_corpus_gen
    if type_corpus_gen == FileCorpusGenerator.TYPE:
        return FileCorpusGenerator.new_instance(FLAGS.file_path)

    if type_corpus_gen == DirCorpusGenerator.TYPE:
        return DirCorpusGenerator.new_instance(FLAGS.dir_path, recursive=True)

    if type_corpus_gen == Wiki2019zhGenerator.TYPE:
        return Wiki2019zhGenerator.new_instance(FLAGS.dir_path, recursive=True)

    else:
        raise ValueError('Unknown corpus iter type: {}'.format(FLAGS.type_corpus_gen))


def build_vocabulary(tokenizer, corpus_iter=None):
    vocab_f_path = FLAGS.vocab_path
    if os.path.exists(vocab_f_path):
        vocab = Vocabulary.new_from_save_file(vocab_f_path, tokenizer=tokenizer)
    else:
        vocab = Vocabulary.new_from_corpus(corpus_iter, tokenizer=tokenizer)
        Vocabulary.save_vocab(vocab, vocab_f_path)
    return vocab


def main(unused_argv):
    del unused_argv  # Unused
    # 语料迭代器
    corpus_iter = build_corpus_iter()

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
    flags.DEFINE_string("dir_path", cfg.DEFAULT_DATA_ROOT, help="Location of the data corpus")
    flags.DEFINE_string("file_path", cfg.DEFAULT_DATA_FILE, help="File of the data corpus")
    flags.DEFINE_string("vocab_path", cfg.DEFAULT_VOCAB_FILE, help="Vocab path where save vocab")
    flags.DEFINE_string("tfrecord_d_path", cfg.DEFAULT_TFRECORDS_D_PATH, help="Tfrecords path where save tfrecords")
    flags.DEFINE_string("record_filename", cfg.DEFAULT_RECORD_FILENAME, help="Record json filename")

    flags.DEFINE_string("type_corpus_gen", cfg.DEFAULT_TYPE_CORPUS_GENERATOR, help="Type of corpus generator")
    flags.DEFINE_string("type_tokenizer", cfg.DEFAULT_TYPE_TOKENIZER, help="Type of corpus generator")

    flags.DEFINE_integer("batch_size", cfg.DEFAULT_BATCH_SIZE, help="train batch size each host")
    flags.DEFINE_integer("tgt_len", cfg.DEFAULT_TGT_LEN, help="number of tokens to predict")
    flags.DEFINE_integer("num_core_per_host", default=cfg.DEFAULT_NUM_CORE, help="Number of cores per host")
    tf.app.run(main)
