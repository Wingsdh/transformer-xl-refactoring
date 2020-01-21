#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :   Wingsdh

@File    :   embedding.py

@Time    :   2020/1/11 15:22

@Desc    :

'''
import tensorflow as tf


def get_mask_adaptive_embedding_lookup_fn(n_token, d_embed, d_proj, initializer, proj_initializer,
                                          scope='adaptive_embed'):
    if proj_initializer is None:
        proj_initializer = initializer

    def mask_adaptive_embedding_lookup(x):

        # d_proj 就是 d_model,
        emb_scale = d_proj ** 0.5
        with tf.variable_scope(scope):
            lookup_table = tf.get_variable('lookup_table', [n_token, d_embed],
                                           initializer=initializer)
            y = tf.nn.embedding_lookup(lookup_table, x)
            # 如果 d_embed与d_model不一样，就无法skip connect了，所以，一旦不一样，强行通过矩阵乘法把词向量的维度弄成d_embed
            if d_proj != d_embed:
                proj_W = tf.get_variable('proj_W', [d_embed, d_proj],
                                         initializer=proj_initializer)
                y = tf.einsum('ibe,ed->ibd', y, proj_W)
            else:
                proj_W = None
            ret_params = [lookup_table, proj_W]

        y *= emb_scale
        return y, ret_params

    return mask_adaptive_embedding_lookup
