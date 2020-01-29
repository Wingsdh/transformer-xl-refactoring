# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    test_text_processor.py
   Description :
   Author :       Wings DH
   Time：         2020/1/22 1:44 下午
-------------------------------------------------
   Change Activity:
                   2020/1/22: Create
-------------------------------------------------
"""
import unittest

from ddt import ddt, data

from data_processing.text_processor import Number2NProcessor, CommonEnglishProcessor, MixProcessor


@ddt
class Number2NProcessorTestCase(unittest.TestCase):
    """
    词库测试用例
    """
    KEY_INP_TEXT = 'inp_text'
    KEY_EXPECT = 'expect'

    @classmethod
    def setUpClass(cls):
        print('Start test Number2NProcessor functions')

    def setUp(self):
        self.text_processor = Number2NProcessor()

    test_data = [
        {
            KEY_INP_TEXT: 'ABC1',
            KEY_EXPECT: 'ABC N'
        },
        {
            KEY_INP_TEXT: 'ABC1A1',
            KEY_EXPECT: 'ABC N A N'
        },
        {
            KEY_INP_TEXT: 'ABC12A',
            KEY_EXPECT: 'ABC NN A'
        },
        {
            KEY_INP_TEXT: 'ABC123A',
            KEY_EXPECT: 'ABC NNN A'
        },
        {
            KEY_INP_TEXT: 'ABC1234A',
            KEY_EXPECT: 'ABC NNNN A'
        },
        {
            KEY_INP_TEXT: 'ABC12345A',
            KEY_EXPECT: 'ABC NNNNN A'
        },
        {
            KEY_INP_TEXT: 'ABC123456A',
            KEY_EXPECT: 'ABC NNNNN A'
        },
    ]

    @data(*test_data)
    def testProcess(self, test_data):
        text = test_data[self.KEY_INP_TEXT]
        expect = test_data[self.KEY_EXPECT]
        result = self.text_processor.process(text)
        self.assertEqual(result, expect)

    @classmethod
    def tearDownClass(cls):
        print('Finish test Number2NProcessor functions')


@ddt
class CommonEnglishProcessorTestCase(unittest.TestCase):
    """
    词库测试用例
    """
    KEY_INP_TEXT = 'inp_text'
    KEY_EXPECT = 'expect'

    @classmethod
    def setUpClass(cls):
        print('Start test CommonEnglishProcessor functions')

    def setUp(self):
        self.text_processor = CommonEnglishProcessor()

    test_data = [
        {
            KEY_INP_TEXT: "i'm here",
            KEY_EXPECT: 'i am here'
        },
        {
            KEY_INP_TEXT: "ABC",
            KEY_EXPECT: 'abc'
        },
    ]

    @data(*test_data)
    def testProcess(self, test_data):
        text = test_data[self.KEY_INP_TEXT]
        expect = test_data[self.KEY_EXPECT]
        result = self.text_processor.process(text)
        self.assertEqual(result, expect)

    @classmethod
    def tearDownClass(cls):
        print('Finish test CommonEnglishProcessor functions')


@ddt
class MixProcessorTestCase(unittest.TestCase):
    """
    词库测试用例
    """
    KEY_INP_TEXT = 'inp_text'
    KEY_EXPECT = 'expect'

    @classmethod
    def setUpClass(cls):
        print('Start test MixProcessor functions')

    def setUp(self):
        num_processor = Number2NProcessor()
        en_processor = CommonEnglishProcessor()

        self.text_processor = MixProcessor([en_processor, num_processor])

    test_data = [
        {
            KEY_INP_TEXT: "i'm here 1",
            KEY_EXPECT: 'i am here  N'
        },
        {
            KEY_INP_TEXT: "AB1C",
            KEY_EXPECT: 'ab N c'
        },
    ]

    @data(*test_data)
    def testProcess(self, test_data):
        text = test_data[self.KEY_INP_TEXT]
        expect = test_data[self.KEY_EXPECT]
        result = self.text_processor.process(text)
        self.assertEqual(result, expect)

    @classmethod
    def tearDownClass(cls):
        print('Finish test MixProcessor functions')


if __name__ == '__main__':
    import os

    os.chdir('../')
    unittest.main()
