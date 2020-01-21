# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    wiki2019zh_generator.py
   Description :
   Author :       Wings DH
   Time：         2020/1/19 9:40 上午
-------------------------------------------------
   Change Activity:
                   2020/1/19: Create
-------------------------------------------------
"""
import json
from corpus_generator.standard_generator import DirCorpusGenerator


class Wiki2019zhGenerator(DirCorpusGenerator):
    TYPE = 'wiki2019zh'

    @staticmethod
    def split_line(line):
        json_string = json.loads(line)
        text = json_string.get('text', '')
        return text.split()
