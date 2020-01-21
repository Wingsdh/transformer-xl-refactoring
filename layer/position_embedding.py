#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :   Wingsdh

@File    :   position_embedding.py

@Time    :   2020/1/11 15:17

@Desc    :

'''
import tensorflow as tf


def positional_embedding(klen, d_model, clamp_len=0, bsz=None):
    pos_seq = tf.range(klen - 1, -1, -1.0)
    if clamp_len > 0:  # pass
        pos_seq = tf.minimum(pos_seq, clamp_len)
    inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))

    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :]
