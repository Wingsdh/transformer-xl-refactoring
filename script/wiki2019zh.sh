#!/bin/bash

# path config
DATASET=wiki2019zh
#DATA_ROOT=/Users/handeng/data/wiki_zh/AA/
#TARGET_ROOT=/Users/handeng/target/wiki2019zh
DATA_ROOT=../../Data/wiki_zh/
TARGET_ROOT=../../target/wiki_zh/
MODEL_ROOT=../../model/wiki_zh/

# tfrecord params
VOCAB_PATH=${TARGET_ROOT}/vocab_wiki2019zh.txt
TFRECORDS_PATH=${TARGET_ROOT}
RECORD_FILENAME=record_info-train.json

# vocab params
TYPE_CORPUS_GEN=wiki2019zh
TYPE_TOKENIZER=char

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

if [[ $1 == 'make_data' ]]; then
  rm -rf ${TARGET_ROOT}/*.tfrecords
  python make_tfrecord.py \
    --dataset=${DATASET} \
    --dir_path=${DATA_ROOT} \
    --vocab_path=${VOCAB_PATH} \
    --tfrecord_d_path=${TFRECORDS_PATH} \
    --type_corpus_gen=${TYPE_CORPUS_GEN} \
    --record_filename=${RECORD_FILENAME} \
    --type_tokenizer=${TYPE_TOKENIZER} \
    --batch_size=${BATCH_SIZE} \
    --tgt_len=${TGT_LEN} \
    "${@:2}"

elif [[ $1 == 'train' ]]; then
  python train.py \
    --dataset=${DATASET} \
    --tfrecord_d_path=${TFRECORDS_PATH} \
    --record_filename=${RECORD_FILENAME} \
    --model_dir=${MODEL_ROOT} \
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
    --train_steps=1000000

  "${@:2}"

else
  echo 'unknown argment 1, must be make_data|train|evaluate'
fi
