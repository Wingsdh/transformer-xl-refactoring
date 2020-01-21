# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    base_generator.py
   Description :
   Author :       Wings DH
   Time：         2020/1/19 9:44 上午
-------------------------------------------------
   Change Activity:
                   2020/1/19: Create
-------------------------------------------------
"""

from abc import ABC, abstractmethod
import os


class ICorpusGenerator(ABC):
    """
    抽象语料迭代器的行为
    """
    TYPE = 'base'

    @classmethod
    @abstractmethod
    def new_instance(cls, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        # 必须实现迭代内置方法
        raise NotImplementedError

    def list_all_file(self, dir_path, recursive=False, check_func=None):
        assert os.path.exists(dir_path)

        paths = os.listdir(dir_path)

        file_names = []
        for p in paths:
            abs_p = os.path.join(dir_path, p)
            if os.path.isdir(abs_p):
                if not recursive:
                    continue

                sub_files = self.list_all_file(abs_p, recursive=recursive, check_func=check_func)
                if len(sub_files) > 0:
                    file_names.extend(sub_files)
            else:
                if check_func is None or check_func(abs_p):
                    file_names.append(abs_p)
        return file_names
