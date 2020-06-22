#!/bin/bash

DATASET=wiki2019zh

# 路径配置
# 训练集源文件目录路径
TRAIN_DATA_ROOT=../../Data/wiki_zh/
TYPE_CORPUS_GEN=wiki2019zh

# 训练集Tfrecord目录路径
TRAIN_TFRECORDS_ROOT=../../TFRecord/${DATASET}/train/
# 验证集源文件目录路径
EVAL_DATA_ROOT=../../Data/wiki_zh_val/
# 验证集Tfrecord目录路径
EVAL_TFRECORDS_ROOT=../../TFRecord/${DATASET}_val/val/
# 模型保存路径
MODEL_ROOT=../../Model/${DATASET}/

# tfrecord params
VOCAB_PATH=./source/spiece.model
RECORD_FILENAME=record_info-train.json


# Model
N_LAYER=12
D_MODEL=768
D_EMBED=768
N_HEAD=12
D_HEAD=64
D_INNER=3072

# train params
BATCH_SIZE=32
TGT_LEN=100
MEM_LEN=100
DROPOUT_RATE=0.1

# eval_params
EVAL_BATCH_SIZE=1

if [[ $1 == 'train_data' ]]; then
  rm -rf ${TRAIN_TFRECORDS_ROOT}/*.tfrecords
  python make_tfrecord.py \
    --dataset=${DATASET} \
    --data_paths=${TRAIN_DATA_ROOT} \
    --type_corpus_gens=${TYPE_CORPUS_GEN} \
    --vocab_path=${VOCAB_PATH} \
    --vocab_type=sentence_piece \
    --tfrecord_d_path=${TRAIN_TFRECORDS_ROOT} \
    --record_filename=${RECORD_FILENAME} \
    --batch_size=${BATCH_SIZE} \
    --tgt_len=${TGT_LEN} \
    --mem_len=${MEM_LEN} \
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

  echo 'unknown argment 1, must be train_data|eval_data|train|evaluate|grade'
fi
