# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    test_vocabulary.py
   Description :
   Author :       Wings DH
   Time：         2020/1/3 2:47 下午
-------------------------------------------------
   Change Activity:
                   2020/1/3: Create
-------------------------------------------------
"""
import os
import shutil
import unittest

from ddt import ddt, data

from data_processing.standard_generator import FileCorpusGenerator
from data_processing.tokenizer import CharTokenizer
from data_processing.vocabulary import Vocabulary
from data_processing.text_processor import Number2NProcessor


@ddt
class VocabularyTestCase(unittest.TestCase):
    """
    词库测试用例
    """
    KEY_NEW_PARAMS = 'params'
    KEY_SAVE_FILE = 'save_file'
    KEY_EXPECT_TOKEN_INDEX = 'expect_t_i'
    KEY_EXPECT_INDEX_TOKEN = 'expect_i_t'
    KEY_TEXT = 'inp_text'
    KEY_EXPECT = 'expect'

    KWARGS_KEY_CORPUS = 'corpus_iter'
    KWARGS_KEY_TOKENIZER = 'tokenizer'
    KWARGS_KEY_TEXT_PROCESSOR = 'text_processor'
    KWARGS_KEY_MAX_N_TOKENS = 'max_n_tokens'
    KWARGS_KEY_MIN_FREQ = 'min_freq'
    KWARGS_KEY_F_PATH = 'f_path'

    TEMP_D_PATH = 'temp/'

    @classmethod
    def setUpClass(cls):
        print('Start test Vocabulary functions')
        if not os.path.exists(cls.TEMP_D_PATH):
            os.mkdir(cls.TEMP_D_PATH)

    def setUp(self):
        pass

    test_data = [
        {
            # 正常输入
            KEY_NEW_PARAMS: {
                KWARGS_KEY_CORPUS: FileCorpusGenerator.new_instance('data/test_corpus/standard/corpus1.txt'),
                KWARGS_KEY_TOKENIZER: CharTokenizer.new_instance()
            },
            KEY_EXPECT_INDEX_TOKEN: {0: Vocabulary.TOKEN_PAD,
                                     1: Vocabulary.TOKEN_UNKNOWN,
                                     2: Vocabulary.TOKEN_START,
                                     3: Vocabulary.TOKEN_END,
                                     4: '一', 5: '二', 6: '三'},
            KEY_EXPECT_TOKEN_INDEX: {Vocabulary.TOKEN_PAD: 0,
                                     Vocabulary.TOKEN_UNKNOWN: 1,
                                     Vocabulary.TOKEN_START: 2,
                                     Vocabulary.TOKEN_END: 3,
                                     '一': 4, '二': 5, '三': 6},
            KEY_SAVE_FILE: 'temp/vocab.txt',
        },
        {
            # 限定最大词数
            KEY_NEW_PARAMS: {
                KWARGS_KEY_CORPUS: FileCorpusGenerator.new_instance('data/test_corpus/standard/corpus1.txt'),
                KWARGS_KEY_TOKENIZER: CharTokenizer.new_instance(),
                KWARGS_KEY_MAX_N_TOKENS: 1,
                KWARGS_KEY_MIN_FREQ: 0,
            },
            KEY_EXPECT_INDEX_TOKEN: {0: Vocabulary.TOKEN_PAD,
                                     1: Vocabulary.TOKEN_UNKNOWN,
                                     2: Vocabulary.TOKEN_START,
                                     3: Vocabulary.TOKEN_END,
                                     4: '一'},
            KEY_EXPECT_TOKEN_INDEX: {Vocabulary.TOKEN_PAD: 0,
                                     Vocabulary.TOKEN_UNKNOWN: 1,
                                     Vocabulary.TOKEN_START: 2,
                                     Vocabulary.TOKEN_END: 3,
                                     '一': 4, },
            KEY_SAVE_FILE: 'temp/vocab.txt',
        },
        {
            # 限定最小词频
            KEY_NEW_PARAMS: {
                KWARGS_KEY_CORPUS: FileCorpusGenerator.new_instance('data/test_corpus/standard/corpus1.txt'),
                KWARGS_KEY_TOKENIZER: CharTokenizer.new_instance(),
                KWARGS_KEY_MIN_FREQ: 2,
            },
            KEY_EXPECT_INDEX_TOKEN: {0: Vocabulary.TOKEN_PAD,
                                     1: Vocabulary.TOKEN_UNKNOWN,
                                     2: Vocabulary.TOKEN_START,
                                     3: Vocabulary.TOKEN_END,
                                     4: '一', 5: '二'},
            KEY_EXPECT_TOKEN_INDEX: {Vocabulary.TOKEN_PAD: 0,
                                     Vocabulary.TOKEN_UNKNOWN: 1,
                                     Vocabulary.TOKEN_START: 2,
                                     Vocabulary.TOKEN_END: 3,
                                     '一': 4, '二': 5},
            KEY_SAVE_FILE: 'temp/vocab.txt',
        },
    ]

    @data(*test_data)
    def testVocabularyBuild(self, test_data):
        """
        测试Vocabulary构造和保存
        :param test_data:
        :return:
        """
        # 从语料加载词库
        params = test_data[self.KEY_NEW_PARAMS]
        vocab = Vocabulary.new_from_corpus(**params)

        expect_token_index = test_data[self.KEY_EXPECT_TOKEN_INDEX]
        expect_index_token = test_data[self.KEY_EXPECT_INDEX_TOKEN]
        self.assertDictEqual(vocab.token_index, expect_token_index)
        self.assertDictEqual(vocab.index_token, expect_index_token)

        # 保存
        target_f_path = test_data[self.KEY_SAVE_FILE]
        Vocabulary.save_vocab(vocab, target_f_path)

        # 重新加载
        vocab = Vocabulary.new_from_save_file(target_f_path)
        self.assertDictEqual(vocab.token_index, expect_token_index)
        self.assertDictEqual(vocab.index_token, expect_index_token)

    test_data_text2seq = [

        {
            # 正常输入
            KEY_NEW_PARAMS: {
                KWARGS_KEY_F_PATH: 'data/vocab.txt',
                KWARGS_KEY_TOKENIZER: CharTokenizer(),
            },
            KEY_TEXT: '一1二2',
            KEY_EXPECT: [4, 1, 5, 1]
        },
        {
            # 增加文本处理器
            KEY_NEW_PARAMS: {
                KWARGS_KEY_F_PATH: 'data/vocab.txt',
                KWARGS_KEY_TOKENIZER: CharTokenizer(),
                KWARGS_KEY_TEXT_PROCESSOR: Number2NProcessor()
            },
            KEY_TEXT: '一1二2',
            KEY_EXPECT: [4, 7, 5, 7]
        }
    ]

    @data(*test_data_text2seq)
    def testVocabularyBuild(self, test_data):
        # 从语料加载词库
        params = test_data[self.KEY_NEW_PARAMS]
        vocab = Vocabulary.new_from_save_file(**params)
        expect = vocab.text_to_sequence(test_data[self.KEY_TEXT])
        self.assertSequenceEqual(expect, test_data[self.KEY_EXPECT])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.TEMP_D_PATH)  # 递归删除文件夹
        print('Finish test Vocabulary functions')


if __name__ == '__main__':
    os.chdir('../')

    unittest.main()
