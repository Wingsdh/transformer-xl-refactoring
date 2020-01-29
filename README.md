# transformer-xl-refactoring
对于论文 [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](http://arxiv.org/abs/1901.02860) 官方源码的重构

## 目的

在项目中用到了transformer-xl，进行了一些工程方面的重构。忙了一年，终于在年底有时间可以整理出来。主要是将自己的理解尽可能的写一些注释，配合知乎文章进行分享，希望可以和更多的朋友一起进步。 

1. 由于模块的划分，以及我个人理解的注释。个人认为对于刚接触的朋友，理解起来会更加友好。
2. 无论是代码还是注释如有错误，也希望得到指正，十分感谢。

## 样例数据集

由于实际项目中使用的是公司数据集，这里选取开源数据集作为样例。代码中采用wiki2019zh数据集，引用自：

 [大规模中文自然语言处理语料 Large Scale Chinese Corpus for NLP](https://github.com/brightmart/nlp_chinese_corpus ) 

## 模块设计思路

1. ### 语料处理模块

   该模块主要目的就是将语料加工处理成需要的模块。

   ICorpusGenerator  <提供统计词频构建词库的语料行迭代> -> Vocabulary

   ITextProcessor <提供文本预处理的方法> -> Vocabulary

   ITextTokenizer <提供将语料行拆分成Token的方法> -> Vocabulary

   Vocabulary <提供将文本行转成索引序列方法> -> TFRecorder

   ICorpusGenerator  <提供语料行迭代方法> -> TFRecorder

   TFRecorder <将语料转存为TFRecord文件>

   ![预处理模块](https://github.com/Wingsdh/transformer-xl-refactoring/raw/master/diagram/corpus_processing.png)



## 参考：

[https://github.com/kimiyoung/transformer-xl](https://github.com/kimiyoung/transformer-xl)

 [大规模中文自然语言处理语料 Large Scale Chinese Corpus for NLP](https://github.com/brightmart/nlp_chinese_corpus ) 

## 