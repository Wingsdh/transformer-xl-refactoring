#!/bin/bash

DATASET=wiki2019zh

# 路径配置
# 训练集源文件目录路径
TRAIN_DATA_ROOT=../../Data/wiki_zh/
# 训练集Tfrecord目录路径
TRAIN_TFRECORDS_ROOT=../../TFRecord/wiki_zh/train/
# 验证集源文件目录路径
EVAL_DATA_ROOT=../../Data/wiki_zh_val/
# 验证集Tfrecord目录路径
EVAL_TFRECORDS_ROOT=../../TFRecord/wiki_zh/val/
# 模型保存路径
MODEL_ROOT=../../Model/wiki_zh/

# tfrecord params
VOCAB_PATH=${TRAIN_TFRECORDS_ROOT}/vocab_wiki2019zh.txt
RECORD_FILENAME=record_info-train.json

# vocab params
TYPE_CORPUS_GEN=wiki2019zh
TYPE_TOKENIZER=char
MIN_FREQ=10
MAX_N_TOKEN=-1

# Model
N_LAYER=3
D_MODEL=100
D_EMBED=100
N_HEAD=3
D_HEAD=6
D_INNER=1000

# train params
BATCH_SIZE=128
TGT_LEN=100
DROPOUT_RATE=0.3

# eval_params
EVAL_BATCH_SIZE=1

if [[ $1 == 'train_data' ]]; then
  rm -rf ${TRAIN_TFRECORDS_ROOT}/*.tfrecords
  python make_tfrecord.py \
    --dataset=${DATASET} \
    --dir_path=${TRAIN_DATA_ROOT} \
    --vocab_path=${VOCAB_PATH} \
    --tfrecord_d_path=${TRAIN_TFRECORDS_ROOT} \
    --type_corpus_gen=${TYPE_CORPUS_GEN} \
    --record_filename=${RECORD_FILENAME} \
    --type_tokenizer=${TYPE_TOKENIZER} \
    --batch_size=${BATCH_SIZE} \
    --tgt_len=${TGT_LEN} \
    --min_freq=${MIN_FREQ} \
    --max_n_token=${MAX_N_TOKEN} \
    "${@:2}"

elif [[ $1 == 'eval_data' ]]; then
  rm -rf ${EVAL_TFRECORDS_ROOT}/*.tfrecords
  python make_tfrecord.py \
    --dataset=${DATASET} \
    --dir_path=${EVAL_DATA_ROOT} \
    --vocab_path=${VOCAB_PATH} \
    --tfrecord_d_path=${EVAL_TFRECORDS_ROOT} \
    --type_corpus_gen=${TYPE_CORPUS_GEN} \
    --record_filename=${RECORD_FILENAME} \
    --type_tokenizer=${TYPE_TOKENIZER} \
    --batch_size=${BATCH_SIZE} \
    --tgt_len=${TGT_LEN} \
    "${@:2}"

elif [[ $1 == 'train' ]]; then
  python train.py \
    --dataset=${DATASET} \
    --tfrecord_d_path=${TRAIN_TFRECORDS_ROOT} \
    --record_filename=${RECORD_FILENAME} \
    --model_dir=${MODEL_ROOT} \
    --warm_start_path=${MODEL_ROOT} \
    --batch_size=${BATCH_SIZE} \
    --tgt_len=${TGT_LEN} \
    --untie_r=True \
    --n_layer=${N_LAYER} \
    --d_model=${D_MODEL} \
    --d_embed=${D_EMBED} \
    --n_head=${N_HEAD} \
    --d_head=${D_HEAD} \
    --d_inner=${D_INNER} \
    --dropout=${DROPOUT_RATE} \
    --dropatt=${DROPOUT_RATE} \
    --learning_rate=0.00010 \
    --warmup_steps=4000 \
    --train_steps=1000000 \
    "${@:2}"
elif [[ $1 == 'evaluate' ]]; then
  python evaluate.py \
    --dataset=${DATASET} \
    --tfrecord_d_path=${EVAL_TFRECORDS_ROOT} \
    --record_filename=${RECORD_FILENAME} \
    --model_dir=${MODEL_ROOT} \
    --warm_start_path=${MODEL_ROOT} \
    --batch_size=${EVAL_BATCH_SIZE} \
    --tgt_len=${TGT_LEN} \
    --untie_r=True \
    --n_layer=${N_LAYER} \
    --d_model=${D_MODEL} \
    --d_embed=${D_EMBED} \
    --n_head=${N_HEAD} \
    --d_head=${D_HEAD} \
    --d_inner=${D_INNER} \
    --dropout=${DROPOUT_RATE} \
    --dropatt=${DROPOUT_RATE} \
    "${@:2}"

else

  echo 'unknown argment 1, must be train_data|eval_data|train|evaluate'
fi
