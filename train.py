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
import math
import os

import numpy as np
import tensorflow as tf
from absl import flags

import common.default_config as cfg
from common.gpu_utils import assign_to_gpu
from common.log import logger

from data_processing.tfrecord_process import TFRecorderLoader
from layer.loss import mask_adaptive_logsoftmax, average_grads_and_vars
from layer.train_op import get_train_op_fn
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
    def build_transformer_xl():
        per_core_bsz = FLAGS.batch_size // FLAGS.num_core_per_host

        transformer_xl = get_transformer_xl_fn(n_token=n_token,
                                               n_layer=FLAGS.n_layer,
                                               d_model=FLAGS.d_model,
                                               d_embed=FLAGS.d_embed,
                                               n_head=FLAGS.n_head,
                                               d_head=FLAGS.d_head,
                                               d_inner=FLAGS.d_inner,
                                               dropout=FLAGS.dropout,
                                               dropatt=FLAGS.dropatt,
                                               initializer=initializer,
                                               proj_initializer=proj_initializer,
                                               is_training=True,
                                               same_length=FLAGS.same_length,
                                               clamp_len=FLAGS.clamp_len,
                                               untie_r=FLAGS.untie_r)
        _tower_mems, tower_losses, _tower_new_mems, tower_grads_and_vars = [], [], [], []

        # assign_to_gpu(i, ps_device)
        for i in range(FLAGS.num_core_per_host):
            inp = tf.transpose(inputs[i], [1, 0])
            target = tf.transpose(labels[i], [1, 0])

            with tf.device(assign_to_gpu(i)), tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                mems = [tf.placeholder(tf.float32, [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                        for _ in range(FLAGS.n_layer)]

                logit, output, new_mems = transformer_xl(inp, mems)

                # loss
                loss_i = mask_adaptive_logsoftmax(logit_output=logit, target=target)

                # 梯度
                _all_vars = tf.trainable_variables()
                _grads = tf.gradients(loss_i, _all_vars)
                grads_and_vars_i = list(zip(_grads, _all_vars))

                _tower_mems.append(mems)
                tower_losses.append(loss_i)
                _tower_new_mems.append(new_mems)
                tower_grads_and_vars.append(grads_and_vars_i)

        #  average losses and gradients across towers
        # 合并所有GPU的loss和梯度
        if len(tower_losses) > 1:
            _loss = tf.add_n(tower_losses) / len(tower_losses)
            _grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
        else:
            _loss = tower_losses[0]
            _grads_and_vars = tower_grads_and_vars[0]

        return _loss, _grads_and_vars, _tower_mems, _tower_new_mems

    loss, grads_and_vars, tower_mems, tower_new_mems, = build_transformer_xl()

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


def train_process(info, tower_mems, tower_new_mems, loss, train_op, learning_rate, gnorm, global_step):
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

        total_loss, prev_step = 0., -1
        init_step = sess.run(global_step)
        epoch = init_step // num_batches
        while True:
            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            fetched = sess.run(fetches, feed_dict=feed_dict)

            loss_np, tower_mems_np, curr_step, gnorm_np, lr_np = fetched[:5]
            total_loss += loss_np

            if curr_step > 0 and curr_step % num_batches == 0:
                epoch += 1
                if curr_step > 0 and epoch % FLAGS.iterations == 0:
                    curr_loss = total_loss / (curr_step - prev_step)
                    logger.info(
                        "[{}] lr {:8.6f} | loss {:.2f} | pplx {:>7.2f} | bpc {:>7.4f} | gnorm {:>7.4f}".format(
                            epoch, lr_np, curr_loss, math.exp(curr_loss), curr_loss / math.log(2), gnorm_np))

                    total_loss, prev_step = 0., curr_step

                if curr_step > 0 and epoch % FLAGS.save_steps == 0:
                    save_path = os.path.join(FLAGS.model_dir, "model-{}.ckpt".format(curr_step))
                    saver.save(sess, save_path)
                    logger.info("Epoch {} Model saved in path: {}".format(epoch, save_path))

            if curr_step == FLAGS.train_steps:
                break

        logger.info("run {} epochs:".format(FLAGS.train_steps // num_batches))


def main(unused_argv):
    del unused_argv  # Unused

    check_paths()

    # 数据
    info, inputs, labels = build_dataset()

    # 训练图
    tower_mems, tower_new_mems, loss, train_op, learning_rate, gnorm, global_step = build_train_graph(inputs, labels,
                                                                                                      info.n_token)

    # 训练
    train_process(info, tower_mems, tower_new_mems, loss, train_op, learning_rate, gnorm, global_step)


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
