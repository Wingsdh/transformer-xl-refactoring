# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    test_tfrecord_process.py
   Description :
   Author :       Wings DH
   Time：         2020/1/8 2:11 下午
-------------------------------------------------
   Change Activity:
                   2020/1/8: Create
-------------------------------------------------
"""
import os
import shutil
import unittest

from ddt import ddt
import tensorflow as tf

from corpus_generator.standard_generator import DirCorpusGenerator
from data_processing.tfrecord_process import TFRecordMaker, TFRecorderLoader
from data_processing.tokenizer import CharTokenizer
from data_processing.vocabulary import Vocabulary


@ddt
class TFRecordProcessTestCase(unittest.TestCase):
    """
    TFRecordMaker 测试用例
    """

    VOCAB_FILE = 'data/vocab.txt'
    TEMP_D_PATH = 'temp'

    BATCH_SIZE = 3
    TGT_LEN = 2
    RECORD_INFO_NAME = 'record_info-train.bsz-{}.tlen-{}.json'.format(BATCH_SIZE, TGT_LEN)

    TEST_DIR_PATH = 'data/test_corpus/standard'
    KEY_KWARGS = 'kwargs'
    KEY_EXPECT = 'expect'

    @classmethod
    def setUpClass(cls):
        print('Start test TFRecordProcessTestCase functions')
        if not os.path.exists(cls.TEMP_D_PATH):
            os.mkdir(cls.TEMP_D_PATH)

    def setUp(self):
        # 预设基本数据
        tokenizer = CharTokenizer.new_instance()
        self.vocab = Vocabulary.new_from_save_file(self.VOCAB_FILE, tokenizer=tokenizer)
        self.line_iter = DirCorpusGenerator.new_instance(self.TEST_DIR_PATH, recursive=True)

    def testTFRecordProcess(self):
        tf_record_maker = TFRecordMaker(self.line_iter, self.vocab, dataset_name='TEST', add_start=True,
                                        add_end=True)

        tf_record_maker.convert_all_tfrecords(self.TEMP_D_PATH, self.BATCH_SIZE, self.TGT_LEN)

        tf_record_loader = TFRecorderLoader(self.TEMP_D_PATH, self.RECORD_INFO_NAME)
        inp_func = tf_record_loader.get_input_fn(TFRecorderLoader.TYPE_TRAIN, self.BATCH_SIZE)
        dataset = inp_func()
        input_feed, label_feed = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            for _ in range(tf_record_loader.info.n_batch):
                _x, _y = sess.run([input_feed, label_feed])
                print('X:{}'.format(self.vocab.sequences_to_texts(_x)))
                print('y:{}'.format(self.vocab.sequences_to_texts(_y)))

        print(tf_record_loader.info)

    @classmethod
    def tearDownClass(cls):
        print('Finish test TFRecordProcessTestCase functions')
        shutil.rmtree(cls.TEMP_D_PATH)


if __name__ == '__main__':
    unittest.main()
