# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    gpu_utils.py
   Description :
   Author :       Wings DH
   Time：         2020/1/20 9:53 上午
-------------------------------------------------
   Change Activity:
                   2020/1/20: Create
-------------------------------------------------
"""

import os
import tensorflow as tf


def assign_to_gpu(gpu=0, ps_dev="/device:CPU:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_dev
        else:
            return "/gpu:%d" % gpu

    return _assign


def load_from_checkpoint(saver, logdir):
    sess = tf.get_default_session()
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
            # Restores from checkpoint with absolute path.
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # Restores from checkpoint with relative path.
            saver.restore(sess, os.path.join(logdir, ckpt.model_checkpoint_path))
        return True
    return False
