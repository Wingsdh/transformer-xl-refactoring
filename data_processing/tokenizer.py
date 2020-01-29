# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    tokenizer.py
   Description :
   Author :       Wings DH
   Time：         2020/1/3 2:48 下午
-------------------------------------------------
   Change Activity:
                   2020/1/3: Create
-------------------------------------------------
"""
from abc import ABC, abstractmethod


class ITextTokenizer(ABC):
    """
    分词器基类
    """

    @classmethod
    @abstractmethod
    def new_instance(cls, *args, **kwargs):
        pass

    @abstractmethod
    def tokenize(self, text):
        """
        Args:
        :type text: str
        :param text: 需要切割的文本
        Returns:tokens: list
        """
        pass


class SpaceTokenizer(ITextTokenizer):
    """
    按照空格分词
    """

    @classmethod
    def new_instance(cls, *args, **kwargs):
        return cls()

    def tokenize(self, text):
        return text.strip().split()


class CharTokenizer(ITextTokenizer):
    """
    按照字符分词
    """

    @classmethod
    def new_instance(cls, *args, **kwargs):
        return cls()

    def tokenize(self, text):
        return [c for c in text if c != ' ']


class CustomTokenizer(ITextTokenizer):
    """
    自定义分词
    """
    ARG_KEY_FUNC = 'func'

    @classmethod
    def new_instance(cls, *args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        func = kwargs.get(cls.ARG_KEY_FUNC, None)
        if func is None:
            raise ValueError('CustomTokenizer can not be None!')
        return cls(func)

    def __init__(self, func):
        self.custom_func = func

    def tokenize(self, text):
        return self.custom_func(text)
