# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    standard_generator.py
   Description :
   Author :       Wings DH
   Time：         2020/1/7 9:07 上午
-------------------------------------------------
   Change Activity:
                   2020/1/7: Create
-------------------------------------------------
"""
import os

from common.log import logger
from corpus_generator.base_generator import ICorpusGenerator


class FileCorpusGenerator(ICorpusGenerator):

    @property
    def f_path(self):
        return self._f_path

    @property
    def encoding(self):
        return self._encoding

    @classmethod
    def new_instance(cls, f_path, encoding='utf-8'):
        """
        单文本语料迭代器工厂方法
        :param f_path: 语料文件路径
        :param encoding: 语料文件编码格式，default:'utf-8'
        :return:SingleFileCorpusGenerator 实例
        """
        if not os.path.exists(f_path):
            raise FileNotFoundError('SingleFileCorpusGenerator f_path:{} must exist '.format(f_path))
        return cls(f_path, encoding)

    def __init__(self, f_path, encoding='utf-8'):
        self._f_path = f_path
        self._encoding = encoding

    def __iter__(self):
        with open(self.f_path, encoding=self._encoding) as f:
            for line in f:
                yield line.strip()


class DirCorpusGenerator(ICorpusGenerator):

    @property
    def d_path(self):
        return self._d_path

    @property
    def encoding(self):
        return self._encoding

    @property
    def recursive(self):
        return self._recursive

    @classmethod
    def new_instance(cls, d_path, encoding='utf-8', recursive=False, check_func=None, split_func=None):
        """
        目录语料迭代器工厂方法
        :param d_path: 语料目录路径
        :param encoding: 语料文件编码格式，default:'utf-8'
        :param recursive: 是否递归子目录，default:False
        :param check_func: 单个文件校验规则 default:None
        :param split_func: 单行语料是否需要拆分 default:None
        :return: DirCorpusGenerator 实例
        """
        if not os.path.exists(d_path):
            raise FileNotFoundError('DirCorpusGenerator d_path:{} must exist '.format(d_path))
        return cls(d_path, encoding, recursive, check_func, split_func)

    def __init__(self, d_path, encoding='utf-8', recursive=False, check_func=None, split_func=None):
        self._d_path = d_path
        self._encoding = encoding
        self._recursive = recursive
        self._check_func = check_func
        self._split_func = split_func

    def __iter__(self):
        file_names = self.list_all_file(self.d_path, recursive=self.recursive, check_func=self._check_func)
        for p in file_names:
            with open(p, 'r', encoding='utf8') as file:
                if self.verbose > 0:
                    logger.info('Start read {}'.format(p))
                for line in file:
                    if self._split_func is not None:
                        for per_yield in self._split_func(line.strip()):
                            yield per_yield
                    else:
                        yield line.strip()
                if self.verbose > 0:
                    logger.info('End  read {}'.format(p))


class MixCorpusGenerator(ICorpusGenerator):

    @classmethod
    def new_instance(cls, its, *args, **kwargs):
        """
        @param its: list, list of ICorpusGenerators
        @param args:
        @param kwargs:
        @return:
        """
        for it in its:
            if not isinstance(it, ICorpusGenerator):
                raise TypeError(
                    'To new a MixCorpusGenerator, it must be instance of {} but {}'.format(ICorpusGenerator, type(it)))
        return cls(its)

    def __init__(self, its):
        self.its = its

    def __iter__(self):
        n_lines = 0
        for it in self.its:
            for line in it:
                if n_lines % 99999 == 0:
                    logger.info('    iter {} lines'.format(n_lines))
                n_lines += 1
                yield line

        logger.info('Finish iter, total {} lines'.format(n_lines))
