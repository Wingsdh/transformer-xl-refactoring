# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    test_tokenizer.py
   Description :
   Author :       Wings DH
   Time：         2020/1/6 5:50 下午
-------------------------------------------------
   Change Activity:
                   2020/1/6: Create
-------------------------------------------------
"""
import unittest

from ddt import ddt, data

from common.log import logger
from data_processing.tokenizer import SpaceTokenizer, CharTokenizer, CustomTokenizer
import jieba


@ddt
class SpaceTokenizerTestCase(unittest.TestCase):
    """
    词库测试用例
    """
    KEY_INP_TEXT = 'inp_text'
    KEY_EXPECT = 'expect'

    @classmethod
    def setUpClass(cls):
        logger.info('Start test SpaceTokenizer functions')

    def setUp(self):
        # 预设基本数据
        self.tokenizer = SpaceTokenizer.new_instance()

    test_data = [
        {
            # 正常输入
            KEY_INP_TEXT: 'A B C',
            KEY_EXPECT: ['A', 'B', 'C']
        },
        {
            # 多个空格
            KEY_INP_TEXT: 'A B  C',
            KEY_EXPECT: ['A', 'B', 'C']
        },
        {
            # 开头结尾空格
            KEY_INP_TEXT: ' A B C ',
            KEY_EXPECT: ['A', 'B', 'C']
        },
        {
            # 空文本
            KEY_INP_TEXT: '',
            KEY_EXPECT: []
        },
    ]

    @data(*test_data)
    def testTokenizer(self, test_data):
        text = test_data[self.KEY_INP_TEXT]
        expect = test_data[self.KEY_EXPECT]
        result = self.tokenizer.tokenize(text)
        self.assertListEqual(result, expect)

    @classmethod
    def tearDownClass(cls):
        logger.info('Finish test SpaceTokenizer functions')


@ddt
class CharTokenizerTestCase(unittest.TestCase):
    """
    词库测试用例
    """
    KEY_INP_TEXT = 'inp_text'
    KEY_EXPECT = 'expect'

    @classmethod
    def setUpClass(cls):
        logger.info('Start test CharTokenizer functions')

    def setUp(self):
        # 预设基本数据
        self.tokenizer = CharTokenizer.new_instance()

    test_data = [
        {
            # 正常输入
            KEY_INP_TEXT: 'A你BC好啊',
            KEY_EXPECT: ['A', '你', 'B', 'C', '好', '啊']
        },
        {
            # 夹杂空格
            KEY_INP_TEXT: 'A 你  BC   好啊',
            KEY_EXPECT: ['A', '你', 'B', 'C', '好', '啊']
        },
        {
            # 空文本
            KEY_INP_TEXT: '',
            KEY_EXPECT: []
        },
    ]

    @data(*test_data)
    def testTokenizer(self, test_data):
        text = test_data[self.KEY_INP_TEXT]
        expect = test_data[self.KEY_EXPECT]
        result = self.tokenizer.tokenize(text)
        self.assertListEqual(result, expect)

    @classmethod
    def tearDownClass(cls):
        logger.info('Finish test CharTokenizer functions')


@ddt
class CustomTokenizerTestCase(unittest.TestCase):
    """
    词库测试用例
    """
    KEY_INP_TEXT = 'inp_text'
    KEY_EXPECT = 'expect'

    @classmethod
    def setUpClass(cls):
        logger.info('Start test CustomTokenizerTestCase functions')

    def setUp(self):
        # 预设基本数据
        def tokenize_func(text):
            return jieba.lcut(text)

        self.tokenizer = CustomTokenizer.new_instance(func=tokenize_func)

    test_data = [
        {
            # 正常输入
            KEY_INP_TEXT: '我爱北京天安门',
            KEY_EXPECT: ['我', '爱', '北京', '天安门']
        },
    ]

    @data(*test_data)
    def testTokenizer(self, test_data):
        """
        测试是否自定义函数是否起作用
        :param test_data:
        :return:
        """
        text = test_data[self.KEY_INP_TEXT]
        expect = test_data[self.KEY_EXPECT]
        result = self.tokenizer.tokenize(text)
        self.assertListEqual(result, expect)

    @classmethod
    def tearDownClass(cls):
        logger.info('Finish test CustomTokenizerTestCase functions')


if __name__ == '__main__':
    import os

    os.chdir('../')
    unittest.main()
