# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    train.py
   Description :
   Author :       Wings DH
   Time：         2020/1/19 11:23 上午
-------------------------------------------------
   Change Activity:
                   2020/1/19: Create
-------------------------------------------------
"""
import os

import numpy as np
import tensorflow as tf
from absl import flags

import common.default_config as cfg
from common.log import logger
from data_processing.tfrecord_process import TFRecorderLoader
from train.train_hook import TimeLogHook, TimeSaverHook
from train.train_op import get_train_op_fn
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
    return tf_record_loader.info, _inputs, _labels


def build_train_graph(inputs, labels, n_token):
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

    grads, all_vars = zip(*grads_and_vars)

    # clip gradient
    # gnorm 就是所有梯度的平方和
    clipped, gnorm = tf.clip_by_global_norm(grads, FLAGS.clip)
    grads_and_vars = list(zip(clipped, all_vars))

    # 已训练步数
    global_step = tf.train.get_or_create_global_step()

    # 定义train_op
    train_fn = get_train_op_fn(FLAGS.train_steps, FLAGS.learning_rate, FLAGS.warmup_steps, FLAGS.min_lr_ratio)
    train_op, learning_rate = train_fn(grads_and_vars=grads_and_vars, global_step=global_step)
    return tower_mems, tower_new_mems, loss, train_op, learning_rate, gnorm, global_step


def train_hooks_builder():
    return [TimeLogHook(60),
            TimeSaverHook(60, model_d_path=FLAGS.model_dir)]


def train_process(info, tower_mems, tower_new_mems, loss, train_op, learning_rate, gnorm, global_step, train_hooks):
    num_batches = info.n_batch
    # Training loop
    per_core_bsz = FLAGS.batch_size // FLAGS.num_core_per_host

    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
         for _ in range(FLAGS.n_layer)] for _ in range(FLAGS.num_core_per_host)
    ]
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        logger.info("Start train transformer-xl lm for dataset:{}".format(FLAGS.dataset))

        if FLAGS.warm_start_path is not None:
            logger.info("warm start from {}".format(FLAGS.warm_start_path))
            try:
                init_ckpt_path = tf.train.latest_checkpoint(FLAGS.warm_start_path)
                saver.restore(sess, init_ckpt_path)
            except ValueError:
                logger.warning('restore fail invalid path:{}'.format(FLAGS.warm_start_path))

        fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op]

        init_step = sess.run(global_step)
        for hook in train_hooks:
            hook.begin(sess=sess, init_step=init_step)
        while True:
            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            for hook in train_hooks:
                hook.before_run()

            fetched = sess.run(fetches, feed_dict=feed_dict)

            loss_np, tower_mems_np, curr_step, gnorm_np, lr_np = fetched[:5]
            for hook in train_hooks:
                hook.after_run(step=curr_step, loss=loss_np, lr=lr_np)

            if curr_step == FLAGS.train_steps:
                for hook in train_hooks:
                    hook.end()
                break

        logger.info("run {} epoch or {} batchs:".format(FLAGS.train_steps // num_batches, curr_step))


def main(unused_argv):
    del unused_argv  # Unused

    check_paths()

    # 数据
    info, inputs, labels = build_dataset()

    # 训练图
    tower_mems, tower_new_mems, loss, train_op, learning_rate, gnorm, global_step = build_train_graph(inputs, labels,
                                                                                                      info.n_token)
    # build hooks
    train_hooks = train_hooks_builder()

    # 训练
    train_process(info, tower_mems, tower_new_mems, loss, train_op, learning_rate, gnorm, global_step, train_hooks)


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
    flags.DEFINE_integer("num_core_per_host", cfg.DEFAULT_NUM_CORE_PER_HOST, help="number of core")

    flags.DEFINE_integer("train_steps", default=cfg.TRAIN_STEPS,
                         help="Total number of training steps.")

    flags.DEFINE_integer("iterations", default=1,
                         help="Number of iterations per repeat loop.")

    flags.DEFINE_integer("save_steps", default=2,
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
