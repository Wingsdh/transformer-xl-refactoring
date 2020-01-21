# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    tfrecord_process.py
   Description :
   Author :       Wings DH
   Time：         2020/1/8 9:12 上午
-------------------------------------------------
   Change Activity:
                   2020/1/8: Create
-------------------------------------------------
"""
import json
import os
import numpy as np
import tensorflow as tf

from common.log import logger
from corpus_generator.standard_generator import ICorpusGenerator
from data_processing.vocabulary import Vocabulary


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


class TFRecordInfo(object):
    KEY_FILE_NAMES = 'filenames'
    KEY_N_BATCH = 'num_batch'
    KEY_N_TOKEN = 'num_token'

    @classmethod
    def new_from_file(cls, d_path, f_name):
        """
        :param d_path:
        :param f_name:
        :return:
        """
        f_path = os.path.join(d_path, f_name)
        if not os.path.exists(f_path):
            raise FileExistsError('TFRecordInfo.new_from_file f_path({}) must exist'.format(f_path))

        info = cls(d_path, f_name)
        info.load_info()

        return info

    @classmethod
    def new_from_data(cls, d_path, f_name, file_names, n_batch, n_token):
        if len(file_names) == 0:
            raise ValueError('TFRecordInfo.new_from_data file_names must not has 0')

        if n_batch <= 0:
            raise ValueError('TFRecordInfo.new_from_data n_batch invalid:{}'.format(n_batch))

        return cls(d_path, f_name, file_names, n_batch, n_token)

    @property
    def n_batch(self):
        return self._n_batch

    @property
    def n_token(self):
        return self._n_token

    @property
    def file_names(self):
        return self._file_names

    @property
    def n_file(self):
        return len(self._file_names)

    @property
    def file_paths(self):
        file_paths = []
        for file_name in self.file_names:
            file_path = os.path.join(self._d_path, file_name)
            file_paths.append(file_path)
        return file_paths

    def __init__(self, d_path, f_name, file_names=None, n_batch=None, n_token=None):
        self._n_batch = n_batch
        self._file_names = file_names
        self._f_name = f_name
        self._d_path = d_path
        self._n_token = n_token

    def load_info(self):
        record_info_f_path = os.path.join(self._d_path, self._f_name)
        with open(record_info_f_path, 'r', encoding='utf-8') as f:
            record_info = json.load(f)
            self._n_batch = record_info[self.KEY_N_BATCH]
            self._file_names = record_info[self.KEY_FILE_NAMES]
            self._n_token = record_info[self.KEY_N_TOKEN]

    def save_info(self):
        record_info_path = os.path.join(self._d_path, self._f_name)
        with open(record_info_path, 'w') as fp:
            record_info = {
                self.KEY_FILE_NAMES: self._file_names,
                self.KEY_N_BATCH: self._n_batch,
                self.KEY_N_TOKEN: self._n_token,
            }
            json.dump(record_info, fp)


class TFRecordMaker(object):
    SINGLE_TFRECORD_SAVE_MAX = 30000000

    def __init__(self, corpus_iter, vocab, dataset_name, add_start=False, add_end=False):
        """
        所有语料需要加工成TFRecord数据格式
        :param corpus_iter:
        :type corpus_iter: ICorpusGenerator
        :param vocab:
        :type vocab: Vocabulary
        """
        if not isinstance(corpus_iter, ICorpusGenerator):
            raise ValueError('corpus_iter must be inst of ICorpusGenerator, but {}'.format(type(ICorpusGenerator)))

        if not isinstance(vocab, Vocabulary):
            raise ValueError('vocab must be inst of Vocabulary, but {}'.format(type(vocab)))

        self._vocab = vocab
        self._corpus_iter = corpus_iter
        self._dataset_name = dataset_name
        self._add_start = add_start
        self._add_end = add_end

    @staticmethod
    def _batchify(data, batch_size):
        num_step = len(data) // batch_size
        data = data[:batch_size * num_step]
        data = data.reshape(batch_size, num_step)
        return data

    @staticmethod
    def _create_ordered_tfrecords(save_d_path, f_name, data, batch_size, tgt_len):
        save_path = os.path.join(save_d_path, f_name)
        record_writer = tf.io.TFRecordWriter(save_path)
        batched_data = TFRecordMaker._batchify(data, batch_size)
        num_batch = 0
        for t in range(0, batched_data.shape[1] - 1, tgt_len):
            cur_tgt_len = min(batched_data.shape[1] - 1 - t, tgt_len)
            if cur_tgt_len < tgt_len:
                break

            if num_batch % 500 == 0:
                logger.debug("  processing batch {}".format(num_batch))
            for idx in range(batch_size):
                inputs = batched_data[idx, t:t + cur_tgt_len]
                labels = batched_data[idx, t + 1:t + cur_tgt_len + 1]

                # features dict
                feature = {
                    "inputs": _int64_feature(inputs),
                    "labels": _int64_feature(labels),
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                record_writer.write(example.SerializeToString())

            num_batch += 1

        record_writer.close()
        logger.debug("Done writing {}. batches: {}".format(f_name, num_batch))

        return f_name, num_batch

    def convert_all_tfrecords(self, save_d_path, batch_size, tgt_len, record_filename=None):
        if not os.path.exists(save_d_path):
            os.makedirs(save_d_path)

        def _check_enough_to_save(total_l_got):
            return total_l_got > TFRecordMaker.SINGLE_TFRECORD_SAVE_MAX

        def _create_tfrecord_process(encoded_data, f_index):
            f_name = "{}_{}.bsz-{}.tlen-{}.tfrecords".format(self._dataset_name, f_index, batch_size, tgt_len)
            data = np.concatenate(encoded_data)
            return self._create_ordered_tfrecords(save_d_path, f_name, data, batch_size, tgt_len)

        encoded = []
        total = 0
        idx_next_file = 0
        total_batch = 0
        file_names = []
        for idx, line in enumerate(self._corpus_iter):
            seq = self._vocab.text_to_sequence(line, add_start=self._add_start, add_end=self._add_end)
            encoded.append(seq)
            total += len(seq)
            if _check_enough_to_save(total):
                filename, n_batch = _create_tfrecord_process(encoded, idx_next_file)
                total_batch += n_batch
                file_names.append(filename)

                encoded = []
                idx_next_file += 1
                total = 0

        filename, n_batch = _create_tfrecord_process(encoded, idx_next_file)
        total_batch += n_batch
        file_names.append(filename)

        record_name = record_filename if record_filename is not None else \
            "record_info-train.bsz-{}.tlen-{}.json".format(batch_size, tgt_len)
        tf_record_info = TFRecordInfo.new_from_data(save_d_path, record_name, file_names, total_batch, len(self._vocab))
        tf_record_info.save_info()


class TFRecorderLoader(object):
    TYPE_TRAIN = 'train'
    TYPE_TEST = 'test'

    @property
    def info(self):
        return self._record_info

    def __init__(self, record_d_path, record_f_name):
        self._record_f_name = record_f_name
        self._record_d_path = record_d_path
        self._record_info = TFRecordInfo.new_from_file(self._record_d_path, self._record_f_name)
        logger.info('Build TFRecorderLoader, batch num:{}, tfrecords num:{}, token num:{}'.format(
            self.info.n_batch, self.info.n_file, self.info.n_token))

    def get_input_fn(self, split, batch_size):
        def _input_fn():
            def parser(record):
                record_spec = {
                    "inputs": tf.VarLenFeature(tf.int64),
                    "labels": tf.VarLenFeature(tf.int64),
                }

                # retrieve serialized example
                example = tf.parse_single_example(
                    serialized=record,
                    features=record_spec)

                # cast int64 into int32
                # cast sparse to dense
                for key in list(example.keys()):
                    val = example[key]
                    if tf.keras.backend.is_sparse(val):
                        # val = tf.sparse.to_dense(val)
                        val = tf.sparse_tensor_to_dense(val)
                    if val.dtype == tf.int64:
                        val = tf.to_int32(val)
                    example[key] = val
                return example["inputs"], example["labels"]

            if split == self.TYPE_TRAIN:
                dataset = tf.data.Dataset.from_tensor_slices(self._record_info.file_paths)
                if self._record_info.n_file > 1:
                    dataset = dataset.shuffle(self._record_info.n_file).repeat()
                    dataset = tf.data.TFRecordDataset(dataset)
                else:
                    dataset = tf.data.TFRecordDataset(self._record_info.file_paths[0])

                dataset = dataset.map(parser).cache().repeat()
                dataset = dataset.batch(batch_size)
                dataset = dataset.prefetch(batch_size)
            else:
                # do not shuffle, repeat or cache in evaluation
                dataset = tf.data.TFRecordDataset(self._record_info.file_paths[0])
                dataset = dataset.map(parser)
                dataset = dataset.batch(batch_size)
            return dataset

        return _input_fn
