# Transformer XL refactoring
对于论文 [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](http://arxiv.org/abs/1901.02860) 官方源码的重构

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tensorflow.svg)]() [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://www.zhihu.com/people/wingsallblue) [![contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Wingsdh/transformer-xl-refactoring/issues) [![GitHub stars](https://img.shields.io/github/stars/wingsdh/transformer-xl-refactoring?style=social)](https://github.com/Wingsdh/transformer-xl-refactoring)



## 目录
- [Transformer XL refactoring](#transformer-xl-refactoring)
  * [目录](#--)
  * [1. 使用](#1---)
    + [1.1下载代码](#11----)
    + [1.2 收集语料](#12-----)
    + [1.3 构建词表](#13-----)
    + [1.4 配置脚本](#14-----)
    + [1.5 语料格式转存为TFRecord](#15--------tfrecord)
    + [1.6 模型训练](#16-----)
    + [1.7 训练监控](#17-----)
    + [1.8 模型部署](#18-----)
  * [2. 重构思路和算法理解分享](#2------------)
  * [3. 参考](#3---)



## 1. 使用

### 1.1下载代码

```bash
git clone https://github.com/Wingsdh/transformer-xl-refactoring
```

### 1.2 收集语料

推荐开源语料：

- [nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus ) 

- [THUCNews](http://thuctc.thunlp.org/#中文文本分类数据集THUCNews)

使用私有语料训练时，需要确定是否用以下存储格式？

```python
class CorpusType(Enum):
    FILE = 'file' # 单个文件
    DIR = 'dir' # 目录下所有txt文件
    WIKI2019 = 'wiki2019zh' # https://github.com/brightmart/nlp_chinese_corpus
```

如果是，可以使用[make_tfrecord.py](https://github.com/Wingsdh/transformer-xl-refactoring/blob/master/make_tfrecord.py)的<data_paths>和<type_corpus_gens>参数传值即可。

如果不是，可以参考[standard_generator.py](https://github.com/Wingsdh/transformer-xl-refactoring/blob/master/data_processing/standard_generator.py)实现一个ICorpusGenerator的子类用于迭代语料。

### 1.3 构建词表

默认使用 SentencePiece 用来实现文本到索引数组的转换，需要根据[官方指导](https://github.com/google/sentencepiece)构建词库文件，并用[make_tfrecord.py](https://github.com/Wingsdh/transformer-xl-refactoring/blob/master/make_tfrecord.py)的<vocab_path>参数传值。

### 1.4 配置脚本

推荐使用脚本来组织训练，参考 [scripts](https://github.com/Wingsdh/transformer-xl-refactoring/tree/master/script)。

PS：支持同时训练多种语料，以','分割即可，比如：

```bash
python make_tfrecord.py \
  --data_paths=../../Data/wiki_zh/,../../Data/THUCNews \
  --type_corpus_gens=wiki2019zh,dir ...
```

### 1.5 语料格式转存为TFRecord

```bash
bash script/wiki2019zh_base.sh train_data
```

### 1.6 模型训练

```bash
bash script/wiki2019zh_base.sh train
```

### 1.7 训练监控

两种监控训练情况的方式：

1. 控制台日志

2. Tensorboard

   ```bash
   tensorboard --logdir=<model_dir>
   ```

### 1.8 模型部署

## 2. 重构思路和算法理解分享

- [重构Transformer-XL代码，关于算法与工程](https://zhuanlan.zhihu.com/p/103769855)
- [NLP任务从语料到 TFRecords 的模块分治](https://zhuanlan.zhihu.com/p/104405967)
- [拆解训练数据如何一层一层“流”过Transformer-XL网络](https://zhuanlan.zhihu.com/p/105472248)

## 3. 参考

- [官方源码](https://github.com/kimiyoung/transformer-xl)
- [大规模中文自然语言处理语料 Large Scale Chinese Corpus for NLP](https://github.com/brightmart/nlp_chinese_corpus ) 
- [THUCNews](http://thuctc.thunlp.org/#中文文本分类数据集THUCNews)
- [SentencePiece](https://github.com/google/sentencepiece)
- [Chinese-XLNet](https://github.com/ymcui/Chinese-XLNet#%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)
