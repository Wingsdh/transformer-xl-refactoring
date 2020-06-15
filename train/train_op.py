# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    train_op.py
   Description :
   Author :       Wings DH
   Time：         2020/1/17 1:57 下午
-------------------------------------------------
   Change Activity:
                   2020/1/17: Create
-------------------------------------------------
"""
import tensorflow as tf


def get_train_op_fn(train_steps, max_learning_rate, warmup_steps, min_lr_ratio):
    def train_op_fn(global_step, grads_and_vars):
        # warmup stage: increase the learning rate linearly
        if warmup_steps > 0:
            warmup_lr = tf.to_float(global_step) / tf.to_float(warmup_steps) \
                        * max_learning_rate
        else:
            warmup_lr = 0.0

        # decay stage: decay the learning rate using the cosine schedule
        decay_lr = tf.train.cosine_decay(
            max_learning_rate,
            global_step=global_step - warmup_steps,
            decay_steps=train_steps - warmup_steps,
            alpha=min_lr_ratio)

        # choose warmup or decay
        learning_rate = tf.where(global_step < warmup_steps,
                                 warmup_lr, decay_lr)

        # get the train op
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

        return train_op, learning_rate

    return train_op_fn
