# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    text_processor.py
   Description :
   Author :       Wings DH
   Time：         2020/1/22 10:26 上午
-------------------------------------------------
   Change Activity:
                   2020/1/22: Create
-------------------------------------------------
"""
from abc import ABC, abstractmethod
import re

from common.log import logger


class ITextProcessor(ABC):

    @abstractmethod
    def process(self, text):
        """
        对text进行预处理，返回处理之后的字符串
        @type text: str
        @return str
        """


class Number2NProcessor(ITextProcessor):
    """
    把文本中的数字全部替换成N
    """
    _rule_1 = re.compile(r'\d{1}')
    _rule_2 = re.compile(r'\d{2}')
    _rule_3 = re.compile(r'\d{3}')
    _rule_4 = re.compile(r'\d{4}')
    _rule_more = re.compile(r'\d{5,}')
    _rules = [_rule_more, _rule_4, _rule_3, _rule_2, _rule_1, ]
    _REPLACE_NUM_1 = ' N '
    _REPLACE_NUM_2 = ' NN '
    _REPLACE_NUM_3 = ' NNN '
    _REPLACE_NUM_4 = ' NNNN '
    _REPLACE_NUM_MORE = ' NNNNN '
    _replaces = [_REPLACE_NUM_MORE, _REPLACE_NUM_4, _REPLACE_NUM_3, _REPLACE_NUM_2, _REPLACE_NUM_1]

    def process(self, text):
        for rule, replace in zip(self._rules, self._replaces):
            text = rule.sub(replace, text)
        return text.strip()


class CommonEnglishProcessor(ITextProcessor):
    """
    常用的英语预处理
    """

    def process(self, text):
        # Lower and token
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        # punction replace
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)  # change to  3 words
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"60k", " 60000 ", text)
        # text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text


class MixProcessor(ITextProcessor):
    def __init__(self, processors):
        """
        多种处理器混合
        处理顺序：按照列表顺序，前面processor输出结果，作为下一个processor输入
        @type processors: list
        @param processors:
        """
        for processor in processors:
            if not isinstance(processor, ITextProcessor):
                raise TypeError('processor must be instance of ITextProcessor, but {}'.format(type(processor)))
        self._processors = processors
        logger.info('Build MixProcessor with {} processors'.format(len(processors)))

    def process(self, text):
        for processor in self._processors:
            text = processor.process(text)
        return text
