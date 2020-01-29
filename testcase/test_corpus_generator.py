# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    test_corpus_generator.py
   Description :
   Author :       Wings DH
   Time：         2020/1/6 5:51 下午
-------------------------------------------------
   Change Activity:
                   2020/1/6: Create
-------------------------------------------------
"""
import unittest

from ddt import ddt, data

from corpus_generator.standard_generator import FileCorpusGenerator, DirCorpusGenerator


class SingleFileCorpusGeneratorTestCase(unittest.TestCase):
    """
    SingleFileCorpusGenerator测试用例
    """
    F_PATH = 'data/test_corpus/standard/corpus1.txt'
    F_ENCODE = 'utf-8'

    @classmethod
    def setUpClass(cls):
        print('Start test SingleFileCorpusGeneratorTestCase functions')

    def setUp(self):
        # 预设基本数据
        self.iter = FileCorpusGenerator.new_instance(f_path=self.F_PATH, encoding=self.F_ENCODE)
        self.expect = {
            0: '一',
            1: '一二',
            2: '一二三',
        }

    def testIter(self):
        """
        测试迭代器是否正确迭代用户数据
        :return:
        """

        for idx, l in enumerate(self.iter):
            self.assertEqual(l, self.expect[idx])

    @classmethod
    def tearDownClass(cls):
        print('Finish test SingleFileCorpusGeneratorTestCase functions')


@ddt
class DirCorpusGeneratorTestCase(unittest.TestCase):
    """
    DirCorpusGenerator测试用例
    """
    TEST_DIR_PATH = 'data/test_corpus/standard/'
    KEY_KWARGS = 'kwargs'
    KEY_EXPECT = 'expect'

    @classmethod
    def setUpClass(cls):
        print('Start test DirCorpusGeneratorTestCase functions')

    test_data = [
        # 递归参数
        {KEY_KWARGS: {'recursive': True},
         KEY_EXPECT: {'一二三四五六七',
                      '一二三四五六七八',
                      '一二三四五六七八九',
                      '一',
                      '一二',
                      '一二三',
                      '一二三四',
                      '一二三四五',
                      '一二三四五六'}},
        {KEY_KWARGS: {'recursive': False},
         KEY_EXPECT: {'一',
                      '一二',
                      '一二三',
                      '一二三四',
                      '一二三四五',
                      '一二三四五六'}}
    ]

    def setUp(self):
        # 预设基本数据
        pass

    @data(*test_data)
    def testIter(self, test_data):
        """
        测试迭代器是否正确迭代用户数据
        """
        kwargs = test_data[self.KEY_KWARGS]
        expect = test_data[self.KEY_EXPECT]
        line_iter = DirCorpusGenerator.new_instance(self.TEST_DIR_PATH, **kwargs)
        result = set()
        for line in line_iter:
            result.add(line)
        self.assertSetEqual(result, expect)

    @classmethod
    def tearDownClass(cls):
        print('Finish test DirCorpusGeneratorTestCase functions')


@ddt
class Wiki2019zhGeneratorTestCase(unittest.TestCase):
    """
    Wiki2019zhGeneratorTestCase 测试用例
    """
    TEST_DIR_PATH = 'data/test_corpus/wiki2019zh/'
    KEY_KWARGS = 'kwargs'
    KEY_EXPECT = 'expect'

    @classmethod
    def setUpClass(cls):
        print('Start test Wiki2019zhGeneratorTestCase functions')

    def setUp(self):
        # 预设基本数据
        pass

    def testIter(self):
        """
        测试迭代器是否正确迭代用户数据
        """
        import json

        def split_line(text):
            json_string = json.loads(text)
            text = json_string.get('text', '')
            return text.split()

        line_iter = DirCorpusGenerator(self.TEST_DIR_PATH, encoding='utf-8', recursive=True, split_func=split_line)
        for idx, line in enumerate(line_iter):
            print("{}:{}".format(idx, line))

    @classmethod
    def tearDownClass(cls):
        print('Finish test Wiki2019zhGeneratorTestCase functions')


if __name__ == '__main__':
    unittest.main()
