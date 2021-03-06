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
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverHook, SummarySaverHook
from tensorflow.python.training.monitored_session import MonitoredSession, ChiefSessionCreator

import common.default_config as cfg
from common.gpu_utils import assign_to_gpu
from common.log import logger
from data_processing.tfrecord_process import TFRecorderLoader
import modeling
from train.loss import average_grads_and_vars
from train.train_hook import TransXlTrainLogHook
from train.train_op import get_train_op_fn


def check_paths():
    # TFRecord 路径必须存在
    if not os.path.exists(FLAGS.tfrecord_d_path):
        raise ValueError('Oh,Oh, TFRecord dir <{}> not exist'.format(FLAGS.dir_path))

    if not os.path.exists(FLAGS.model_dir):
        logger.info('Model dir <{}> not exist, create it'.format(FLAGS.model_dir))
        os.makedirs(FLAGS.model_dir)

    log_path = os.path.join(FLAGS.model_dir, 'logs')
    if not os.path.exists(log_path):
        logger.info('Log dir <{}> not exist, create it'.format(log_path))
        os.makedirs(log_path)

    ckpt_path = os.path.join(FLAGS.model_dir, 'ckpt')
    if not os.path.exists(ckpt_path):
        logger.info('checkpoint dir <{}> not exist, create it'.format(ckpt_path))
        os.makedirs(ckpt_path)
    return FLAGS.tfrecord_d_path, log_path, ckpt_path


def build_dataset(tfrecord_d_path):
    tf_record_loader = TFRecorderLoader(tfrecord_d_path, FLAGS.record_filename)
    inp_func = tf_record_loader.get_input_fn(TFRecorderLoader.TYPE_TRAIN, FLAGS.batch_size)
    dataset = inp_func()
    input_feed, label_feed = dataset.make_one_shot_iterator().get_next()
    _inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
    _labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)
    return tf_record_loader.info, _inputs, _labels


def get_model_fn(n_token, cutoffs):
    def model_fn(inp, tgt, mems, is_training):
        inp = tf.transpose(inp, [1, 0])
        tgt = tf.transpose(tgt, [1, 0])

        if FLAGS.init == "uniform":
            initializer = tf.initializers.random_uniform(
                minval=-FLAGS.init_range,
                maxval=FLAGS.init_range,
                seed=None)
        elif FLAGS.init == "normal":
            initializer = tf.initializers.random_normal(
                stddev=FLAGS.init_std,
                seed=None)
            proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std,
                seed=None)

        tie_projs = [False for _ in range(len(cutoffs) + 1)]

        loss, new_mems = modeling.transformer(
            dec_inp=inp,
            target=tgt,
            mems=mems,
            n_token=n_token,
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
            is_training=is_training,
            mem_len=FLAGS.mem_len,
            cutoffs=cutoffs,
            tie_projs=tie_projs,
            input_perms=None,
            target_perms=None,
            head_target=None,
            same_length=FLAGS.same_length,
            clamp_len=FLAGS.clamp_len,
            use_tpu=False,
            untie_r=FLAGS.untie_r)

        # number of parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        logger.info('#params: {}'.format(num_params))

        if is_training:
            all_vars = tf.trainable_variables()
            grads = tf.gradients(loss, all_vars)
            grads_and_vars = list(zip(grads, all_vars))

            return loss, new_mems, grads_and_vars
        else:
            return loss, new_mems

    return model_fn


def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems):
    model_fn = get_model_fn(
        n_token=n_token,
        cutoffs=cutoffs)

    model_ret = model_fn(
        inp=inp,
        tgt=tgt,
        mems=mems,
        is_training=is_training)

    return model_ret


def build_train_graph(inputs, labels, n_token, ps_device, cutoffs):
    per_core_bsz = FLAGS.batch_size // FLAGS.num_core_per_host

    tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

    for i in range(FLAGS.num_core_per_host):
        reuse = True if i > 0 else None
        with tf.device(assign_to_gpu(i, ps_device)), tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            mems_i = [tf.placeholder(tf.float32,
                                     [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]

            loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
                n_token=n_token,
                cutoffs=cutoffs,
                is_training=True,
                inp=inputs[i],
                tgt=labels[i],
                mems=mems_i)

            tower_mems.append(mems_i)
            tower_losses.append(loss_i)
            tower_new_mems.append(new_mems_i)
            tower_grads_and_vars.append(grads_and_vars_i)

    # average losses and gradients across towers
    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
        grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
    else:
        loss = tower_losses[0]
        grads_and_vars = tower_grads_and_vars[0]
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

    def build_summary_op():
        with tf.name_scope('summary'):
            tf.summary.scalar('learn_rate', learning_rate)
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('pplx', tf.exp(loss))
            tf.summary.scalar('bpc', loss / math.log(2))
        return tf.summary.merge_all()

    summary_op = build_summary_op()

    return tower_mems, tower_new_mems, loss, train_op, learning_rate, gnorm, global_step, summary_op


def train_hooks_builder(loss, global_step, learn_rate, summary_op, log_dir, ckpt_dir):
    log_hook = TransXlTrainLogHook(loss_tensor=loss,
                                   learn_rate_tensor=learn_rate,
                                   step_tensor=global_step,
                                   every_n_iter=FLAGS.iterations)

    summary_hook = SummarySaverHook(1, summary_op=summary_op, output_dir=log_dir)
    ckpt_save_hook = CheckpointSaverHook(checkpoint_dir=ckpt_dir, save_steps=FLAGS.save_steps)
    return [log_hook, ckpt_save_hook, summary_hook]


def train_process(tower_mems, tower_new_mems, train_op, train_hooks, ckpt_dir):
    # Training loop
    per_core_bsz = FLAGS.batch_size // FLAGS.num_core_per_host

    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
         for _ in range(FLAGS.n_layer)] for _ in range(FLAGS.num_core_per_host)
    ]

    logger.info("Start train transformer-xl lm for dataset:{}".format(FLAGS.dataset))

    with MonitoredSession(session_creator=ChiefSessionCreator(config=tf.ConfigProto(allow_soft_placement=True),
                                                              checkpoint_dir=ckpt_dir),
                          hooks=train_hooks) as sess:

        fetches = [tower_new_mems, train_op]
        feed_dict = {}
        while not sess.should_stop():
            # Segment - Level Recurrence with State Reuse
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            fetched = sess.run(fetches, feed_dict=feed_dict)
            tower_mems_np, _ = fetched


def main(unused_argv):
    del unused_argv  # Unused

    tfrecord_d_path, log_path, ckpt_path = check_paths()

    info, inputs, labels = build_dataset(tfrecord_d_path)

    # train graph
    tower_mems, tower_new_mems, loss, train_op, learning_rate, gnorm, global_step, summary_op = build_train_graph(
        inputs, labels,
        info.n_token, '/gpu:0', [])

    # build hooks
    train_hooks = train_hooks_builder(loss, global_step, learning_rate, summary_op, log_path, ckpt_path)

    train_process(tower_mems, tower_new_mems, train_op, train_hooks, ckpt_path)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

    flags.DEFINE_integer("iterations", default=cfg.N_LOG_STEPS,
                         help="Number of iterations per repeat loop.")
    flags.DEFINE_integer("save_steps", default=cfg.N_SAVE_STEPS,
                         help="number of steps for model checkpointing.")
    flags.DEFINE_integer("summary_steps", default=cfg.N_SAVE_SUMMARY_STEPS,
                         help="number of steps for save summary.")
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
