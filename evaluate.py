# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    evaluate.py
   Description :
   Author :       Wings DH
   Time：         2020/1/21 2:08 下午
-------------------------------------------------
   Change Activity:
                   2020/1/21: Create
-------------------------------------------------
"""
import math
import os
import sys

import tensorflow as tf
from absl import flags
import numpy as np

import common.default_config as cfg
from common.log import logger
from data_processing.tfrecord_process import TFRecorderLoader
from model.transformer_xl import build_transformer_xl


def check_paths():
    # TFRecord 路径必须存在
    if not os.path.exists(FLAGS.tfrecord_d_path):
        raise ValueError('Oh,Oh, TFRecord dir <{}> not exist'.format(FLAGS.dir_path))

    if not os.path.exists(FLAGS.model_dir):
        logger.info('Model dir <{}> not exist, create it'.format(FLAGS.model_dir))
        os.makedirs(FLAGS.model_dir)


def build_dataset():
    tf_record_loader = TFRecorderLoader(FLAGS.tfrecord_d_path, FLAGS.record_filename)
    inp_func = tf_record_loader.get_input_fn(TFRecorderLoader.TYPE_TRAIN, FLAGS.batch_size)
    dataset = inp_func()
    input_feed, label_feed = dataset.make_one_shot_iterator().get_next()
    _inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
    _labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)
    return tf_record_loader.info, _inputs, _labels, label_feed


def build_eval_graph(inputs, labels, n_token):
    def build_initializer():
        _initializer = None
        _proj_initializer = None
        if FLAGS.init == "uniform":
            _initializer = tf.initializers.random_uniform(
                minval=-FLAGS.init_range,
                maxval=FLAGS.init_range,
                seed=None)
        elif FLAGS.init == "normal":  # select
            _initializer = tf.initializers.random_normal(
                stddev=FLAGS.init_std,
                seed=None)
            _proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std,
                seed=None)
        return _initializer, _proj_initializer

    initializer, proj_initializer = build_initializer()

    # 定义 transformer 骨干网络
    loss, grads_and_vars, tower_mems, tower_new_mems, = build_transformer_xl(inputs, labels, FLAGS, n_token,
                                                                             initializer,
                                                                             proj_initializer)

    return loss, tower_mems, tower_new_mems


def evaluate_process(label_feed, info, tower_mems, tower_new_mems, loss):
    num_batches = info.n_batch
    logger.info('Eval Data has {} batches'.format(num_batches))

    # Evaluate
    per_core_bsz = FLAGS.batch_size // FLAGS.num_core_per_host

    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
         for _ in range(FLAGS.n_layer)] for _ in range(FLAGS.num_core_per_host)
    ]
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.eval_ckpt_path is None:
            eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
        else:
            eval_ckpt_path = FLAGS.eval_ckpt_path

        logger.info("Evaluate {}".format(eval_ckpt_path))
        saver.restore(sess, eval_ckpt_path)

        fetches = [loss, tower_new_mems, tf.size(label_feed)]

        format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(len(str(num_batches)))

        total_loss, total_cnt = 0, 0
        for step in range(num_batches):
            if step % (num_batches // 10) == 0:
                logger.info(format_str.format(step, num_batches))

            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            fetched = sess.run(fetches, feed_dict=feed_dict)

            loss_np, tower_mems_np, cnt_np = fetched[:3]
            total_loss += loss_np * cnt_np
            total_cnt += cnt_np

        avg_loss = total_loss / total_cnt
        logger.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
            avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))


def main(argv=None):
    if argv is None:
        argv = sys.argv

    check_paths()

    # 数据
    info, inputs, labels, label_feed = build_dataset()

    # 构建图
    loss, tower_mems, tower_new_mems = build_eval_graph(inputs, labels, info.n_token)

    # 评估过程
    evaluate_process(label_feed, info, tower_mems, tower_new_mems, loss)


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_string("dataset", default=cfg.DEFAULT_DATASET, help="Dataset name.")
    flags.DEFINE_string("tfrecord_d_path", cfg.DEFAULT_TFRECORDS_D_PATH, help="Tfrecords path where save tfrecords")
    flags.DEFINE_string("record_filename", cfg.DEFAULT_RECORD_FILENAME, help="Record json filename")

    flags.DEFINE_integer("batch_size", cfg.DEFAULT_BATCH_SIZE, help="train batch size each host")
    flags.DEFINE_integer("tgt_len", cfg.DEFAULT_TGT_LEN, help="number of tokens to predict")
    flags.DEFINE_integer("mem_len", default=cfg.MEM_LEN,
                         help="Number of steps to cache")

    # Train
    flags.DEFINE_string("warm_start_path", None,
                        help="Checkpoint path for warm start."
                             "If set, will clear Adam states."
                             "Note that the new model_dir should be different"
                             " from warm_start_path.")

    flags.DEFINE_string("eval_ckpt_path", None,
                        help="Checkpoint path for do_test evaluation."
                             "If set, model_dir will be ignored."
                             "If unset, will use the latest ckpt in model_dir.")
    flags.DEFINE_integer("num_core_per_host", cfg.DEFAULT_NUM_CORE_PER_HOST, help="number of core")

    flags.DEFINE_integer("train_steps", default=cfg.TRAIN_STEPS,
                         help="Total number of training steps.")

    flags.DEFINE_integer("iterations", default=2,
                         help="Number of iterations per repeat loop.")

    flags.DEFINE_integer("save_steps", default=4,
                         help="number of steps for model checkpointing.")
    flags.DEFINE_string("model_dir", default=None,
                        help="Estimator model_dir.")

    # Optimization config
    flags.DEFINE_float("learning_rate", default=0.00025,
                       help="Maximum learning rate.")
    flags.DEFINE_float("clip", default=0.25,
                       help="Gradient clipping value.")

    # for cosine decay
    flags.DEFINE_float("min_lr_ratio", default=0.004,
                       help="Minimum ratio learning rate.")
    flags.DEFINE_integer("warmup_steps", default=0,
                         help="Number of steps for linear lr warmup.")

    # Model

    flags.DEFINE_bool("same_length", default=False,
                      help="Same length attention")
    flags.DEFINE_integer("clamp_len", default=-1,
                         help="Clamp length")

    flags.DEFINE_integer("n_layer", default=cfg.N_LAYER,
                         help="Number of layers.")
    flags.DEFINE_integer("d_model", default=cfg.D_MODEL,
                         help="Dimension of the model.")
    flags.DEFINE_integer("d_embed", default=cfg.D_EMBED,
                         help="Dimension of the embeddings.")
    flags.DEFINE_integer("n_head", default=cfg.N_HEAD,
                         help="Number of attention heads.")
    flags.DEFINE_integer("d_head", default=cfg.D_HEAD,
                         help="Dimension of each attention head.")
    flags.DEFINE_integer("d_inner", default=cfg.D_INNER,
                         help="Dimension of inner hidden size in positionwise feed-forward.")
    flags.DEFINE_float("dropout", default=cfg.DROPOUT,
                       help="Dropout rate.")
    flags.DEFINE_float("dropatt", default=cfg.DROPOUT,
                       help="Attention dropout rate.")
    flags.DEFINE_bool("untie_r", default=False,
                      help="untie r_w_bias and r_r_bias")

    # Parameter initialization
    flags.DEFINE_enum("init", default="normal",
                      enum_values=["normal", "uniform"],
                      help="Initialization method.")
    flags.DEFINE_float("init_std", default=0.02,
                       help="Initialization std when init is normal.")
    flags.DEFINE_float("proj_init_std", default=0.01,
                       help="Initialization std for embedding projection.")
    flags.DEFINE_float("init_range", default=0.1,
                       help="Initialization std when init is uniform.")

    tf.app.run(main)
