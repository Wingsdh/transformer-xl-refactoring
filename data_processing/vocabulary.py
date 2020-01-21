# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    vocabulary.py
   Description :
   Author :       Wings DH
   Time：         2020/1/4 2:47 下午
-------------------------------------------------
   Change Activity:
                   2020/1/3: Create
-------------------------------------------------
"""
from abc import ABC
from collections import OrderedDict

from tqdm import tqdm

from common.log import logger
from data_processing.tokenizer import SpaceTokenizer, ITextTokenizer


def _save_vocab_to_file(vocab, save_file_path):
    """ 保存词库到txt文件
    :param save_file_path:
    :param vocab:
    Returns:
    """
    if save_file_path is not None:
        with open(save_file_path, 'w', encoding='utf-8') as f:
            for index in vocab.index_token:
                _token = vocab.index_token[index]
                f.write('{token}:{index}\n'.format(token=_token, index=index))
            logger.info('Save vocab into {vocab_file}'.format(vocab_file=save_file_path))


def _arrange_vocab_from_corpus(corpus_iter, tokenizer):
    """
    从语料中获取词库
    :param corpus_iter: 关于corpus的迭代器, 每次迭代输出一个token列表
    :return vocab: 构建好的Vocabulary
    """

    if not corpus_iter:
        raise ValueError('file_path and dir_path cant be None in the same time!')

    if not tokenizer:
        tokenizer = SpaceTokenizer.new_instance()
    elif not isinstance(tokenizer, ITextTokenizer):
        raise ValueError('tokenizer:<{}> must instance of ITextTokenizer'.format(type(tokenizer)))

    vocab = Vocabulary(tokenizer=tokenizer)
    for line in tqdm(corpus_iter):

        tokens = tokenizer.tokenize(line)

        for token in tokens:
            vocab.count_token(token)

    vocab.build_vocab_from_word_count()

    return vocab


def _load_from_token_count_txt(path, tokenizer=None):
    vocab = Vocabulary(tokenizer=tokenizer)
    try:
        logger.info('Start load token_count txt')
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    coef = line.strip().split(':')

                    # 防止token=:的情况
                    _token = coef[0]
                    _index = coef[-1]

                    vocab.token_index[_token] = int(_index)
                    vocab.index_token[int(_index)] = _token
                except ValueError:
                    logger.warning('Invalid line:{}'.format(line))

        logger.info('End   load token_count txt, read {nb_tokens} tokens'.format(nb_tokens=len(vocab)))
    except FileNotFoundError:
        logger.warning(
            'File not exit, please check {error_file}'.format(error_file=path))
    return vocab


class Vocabulary(ABC):
    """
    词汇库类
    """
    TOKEN_PAD = '<pad>'
    TOKEN_UNKNOWN = '<unk>'
    TOKEN_START = '<s>'
    TOKEN_END = '</s>'

    INDEX_PAD = 0
    INDEX_UNKNOWN = 1
    INDEX_START = 2
    INDEX_END = 3
    INDEX_ORIGIN_CUSTOM = 4

    NO_LIMIT_MAX_N_WORD = -1

    @property
    def index_token(self):
        return self._index_token

    @property
    def token_index(self):
        return self._token_index

    @property
    def token_count(self):
        return self._token_count

    @classmethod
    def new_from_corpus(cls, corpus_iter, tokenizer):
        return _arrange_vocab_from_corpus(corpus_iter, tokenizer)

    @classmethod
    def new_from_save_file(cls, f_path, tokenizer=None):
        return _load_from_token_count_txt(f_path, tokenizer)

    @staticmethod
    def save_vocab(vocab, f_path):
        """
        保存vocab到文件
        :param vocab: 需要保存的Vocabulary实例
        :param f_path: 保存路径
        :return:
        """
        _save_vocab_to_file(vocab, f_path)

    def __init__(self, tokenizer=None):

        # 初始化词库
        self._token_index = {}
        self._index_token = {}
        self._token_count = OrderedDict()

        # 初始化分词器
        if tokenizer is None:
            # 默认使用空格来做分词
            self.tokenizer = SpaceTokenizer.new_instance()
        else:
            self.tokenizer = tokenizer

    def count_token(self, token):
        if token in self.token_count:
            self.token_count[token] += 1
        else:
            self.token_count[token] = 1

    def add_token_index(self, token, index):
        self._token_index[token] = index
        self._index_token[index] = token

    def build_vocab_from_word_count(self, max_n_tokens=NO_LIMIT_MAX_N_WORD):
        self._token_index = {}
        self._index_token = {}

        # 默认占前4位索引
        self.add_token_index(Vocabulary.TOKEN_PAD, Vocabulary.INDEX_PAD)
        self.add_token_index(Vocabulary.TOKEN_UNKNOWN, Vocabulary.INDEX_UNKNOWN)
        self.add_token_index(Vocabulary.TOKEN_START, Vocabulary.INDEX_START)
        self.add_token_index(Vocabulary.TOKEN_END, Vocabulary.INDEX_END)

        index = Vocabulary.INDEX_ORIGIN_CUSTOM

        # 根据词频排序，只保留高频词
        wcounts = list(self.token_count.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)

        for cnt, (w, c) in enumerate(wcounts):
            if max_n_tokens != self.NO_LIMIT_MAX_N_WORD and cnt > max_n_tokens:
                break
            self.add_token_index(w, index)
            index += 1
        logger.info('Build vocab, volume:{size_word_index}'.format(size_word_index=len(self.token_index)))

    def tokens_to_sequence(self, tokens, add_start=False, add_end=False):
        """
        将一句话的token转成序列
        :param tokens:
        :param add_start:
        :param add_end:
        :return:
        """
        sequence = []
        if add_start:
            sequence.append(self.token_index[Vocabulary.TOKEN_START])

        sequence.extend([self.token_index.get(c, self.token_index[Vocabulary.TOKEN_UNKNOWN]) for c in tokens])

        if add_end:
            sequence.append(self.token_index[Vocabulary.TOKEN_END])

        return sequence

    def text_to_sequence(self, text, add_start=False, add_end=False):
        """
        文本转成序列
        :param text:
        :param add_start:
        :param add_end:
        :return:
        """
        # 先用分词器将一段文本转成tokens
        tokens = self.tokenizer.tokenize(text)
        return self.tokens_to_sequence(tokens, add_start=add_start, add_end=add_end)

    def texts_to_sequences(self, texts, add_start=False, add_end=False):
        """
        多个文本转成序列
        :param texts:
        :param add_start:
        :param add_end:
        :return:
        """
        sequences = []
        for text in texts:
            seq = self.text_to_sequence(text, add_start=add_start, add_end=add_end)
            sequences.append(seq)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for seq in sequences:
            texts.append([self.index_token.get(token, Vocabulary.TOKEN_UNKNOWN) for token in seq])
        return texts

    def __len__(self):
        return len(self._token_index)

    def __contains__(self, item):
        return item in self._token_index

    def __getitem__(self, item):
        return self.token_index.get(item, self.token_index[Vocabulary.TOKEN_UNKNOWN])

    def __str__(self):
        """
        Returns: summary str
        """
        return 'Word Total: {total}'.format(total=len(self))
