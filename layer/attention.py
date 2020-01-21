#!/usr/bin/env python

# -*- encoding: utf-8 -*-

"""
@Author  :   Wingsdh
@File    :   attention.py
@Time    :   2020/1/11 15:18
@Desc    :
'''
"""
import tensorflow as tf


def _create_mask(qlen, mlen, same_length=False):
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.matrix_band_part(attn_mask, 0, -1)
    mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


def _rel_shift(x):
    x_size = tf.shape(x)

    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)

    return x


def build_rel_multihead_attn_func(qlen, mlen, n_layer, d_model, n_head, d_head, dropout, dropatt, initializer,
                                  is_training, same_length=False, untie_r=False):
    # 关于是否采用参数共享：
    # Transformer 中各层attention参数不共享
    # Universal Transformer 中采用参数共享，据说是共享之后可以得到更稳定的梯度
    # 我在我的任务中测试是不共享能得到更好的准确度。还是以测试为准吧。
    # ToDo：如果阅读到确切的结论或研究之后，更新这里的注释。
    if untie_r:
        r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head],
                                   initializer=initializer)
        r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                   initializer=initializer)
    else:  # select
        r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
                                   initializer=initializer)
        r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
                                   initializer=initializer)

    # 关于为什么先缩放后做 softmax，最早我是在  <Attention is all you need> 计算公式总看到
    # 后查阅资料学习主要是两点：
    # 1. 理论角度 softmax 在较大值上曲线趋于平缓，产生梯度消失，所以按尺寸缩小能让梯度更加稳定。
    # 2. 至于为什么用d_head ** 0.5来缩放，就是实验结果。没有严格的理论证明，只是实验得到这样还不错。
    # 参考链接
    # https://arxiv.org/abs/1706.03762
    # //jalammar.github.io/illustrated-transformer/
    # https://www.zhihu.com/question/339723385/answer/782509914
    scale = 1 / (d_head ** 0.5)

    attn_mask = _create_mask(qlen, mlen, same_length)

    def rel_multihead_attn(index, rel_inp, w, mem):
        rlen = tf.shape(rel_inp)[0]

        qlen = tf.shape(w)[0]
        bsz = tf.shape(w)[1]

        if mem is not None and mem.shape.ndims > 1:
            cat = tf.concat([mem, w], 0)
        else:
            cat = w

        w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False,
                                  kernel_initializer=initializer, name='qkv')

        # 残差连接
        r_head_k = tf.layers.dense(rel_inp, n_head * d_head, use_bias=False,
                                   kernel_initializer=initializer, name='r')

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

        r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

        _r_w_bias = r_w_bias if not untie_r else r_w_bias[index]
        rw_head_q = w_head_q + _r_w_bias

        _r_r_bias = r_r_bias if not untie_r else r_r_bias[index],
        rr_head_q = w_head_q + _r_r_bias

        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = _rel_shift(BD)

        attn_score = (AC + BD) * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                                   kernel_initializer=initializer, name='o')
        attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

        # ToDo: tf 1.x 是否有tf.contrib 之外的接口替代方案？
        output = tf.contrib.layers.layer_norm(attn_out + w, begin_norm_axis=-1)
        return output

    return rel_multihead_attn
