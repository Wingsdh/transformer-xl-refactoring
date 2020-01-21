# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    transformer_xl.py
   Description :
   Author :       Wings DH
   Time：         2020/1/11 5:51 下午
-------------------------------------------------
   Change Activity:
                   2020/1/11: Create
-------------------------------------------------
"""

import tensorflow as tf

from common.gpu_utils import assign_to_gpu
from layer.attention import build_rel_multihead_attn_func
from layer.embedding import get_mask_adaptive_embedding_lookup_fn
from layer.loss import average_grads_and_vars, mask_adaptive_logsoftmax
from layer.position_embedding import positional_embedding


def _cache_mem(curr_out, prev_mem, mem_len=None):
    if mem_len is None or prev_mem is None:
        new_mem = curr_out
    elif mem_len == 0:
        return prev_mem
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[- mem_len:]

    return tf.stop_gradient(new_mem)


def get_positionwise_feed_forward_fn(d_model, d_inner, dropout, kernel_initializer,
                                     scope='ff', is_training=True):
    def positionwise_feed_forward(inp):
        with tf.variable_scope(scope):
            output = tf.layers.dense(inp, d_inner, activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     name='layer_1')
            output = tf.layers.dropout(output, dropout, training=is_training,
                                       name='drop_1')
            output = tf.layers.dense(output, d_model,
                                     kernel_initializer=kernel_initializer,
                                     name='layer_2')
            output = tf.layers.dropout(output, dropout, training=is_training,
                                       name='drop_2')
            output = tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1)
        return output

    return positionwise_feed_forward


def logit_output(hidden, n_token, params, scope='adaptive_softmax'):
    def _logit(x, W, b, proj):
        y = x
        if proj is not None:
            y = tf.einsum('ibd,ed->ibe', y, proj)
        return tf.einsum('ibd,nd->ibn', y, W) + b

    params_W, params_projs = params[0], params[1]

    with tf.variable_scope(scope):
        softmax_b = tf.get_variable('bias', [n_token],
                                    initializer=tf.zeros_initializer())
        output = _logit(hidden, params_W, softmax_b, params_projs)
    return output


def get_transformer_xl_fn(n_token, n_layer, d_model, d_embed,
                          n_head, d_head, d_inner, dropout, dropatt, initializer,
                          is_training, proj_initializer=None,
                          mem_len=None, same_length=False, clamp_len=-1,
                          untie_r=False, scope='transformer'):
    """
    transformer xl 核心网络
    :param n_token: 词库大小
    :param n_layer: rel_multihead_attn 层数
    :param d_model:
    :param d_embed: 词嵌入维度，第一层 rel_multihead_attn 输入
    :param n_head: 每一层 rel_multihead_attn 并列的"头"数
    :param d_head:
    :param d_inner:
    :param dropout: 训练中 由于 dropout 失效的参数比例
    :param dropatt: 只用于attention计算权重之后的dropout
    :param initializer: 参数处事方式
    :param is_training: 用于给dropout提供是否起作用的flag，直接改成除训练模式直接不建立dropout层
    :param proj_initializer: 映射层初始化方式，None的话直接等同于initializer
    :param mem_len:
    :param same_length:
    :param clamp_len:
    :param untie_r: 所有rel_multihead_attn是否共享参数
    :param scope:
    :return:
    """

    def transformer_xl(inp, mems):
        """
        :param inp: 输入
        :param mems: 记忆单元，上一个batch数据得到，每个batch训练开始前更新
        :return:
        """
        new_mems = []

        qlen = tf.shape(inp)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen

        # 定义词嵌入层
        lookup_fn = get_mask_adaptive_embedding_lookup_fn(
            n_token=n_token,
            d_embed=d_embed,
            d_proj=d_model,
            initializer=initializer,
            proj_initializer=proj_initializer,
        )

        # 定义 rel_multihead_attn 层
        attn_layer_func = build_rel_multihead_attn_func(qlen, mlen, n_layer, d_model, n_head,
                                                        d_head, dropout, dropatt, initializer,
                                                        is_training, same_length=same_length,
                                                        untie_r=untie_r)

        # 定义 positionwise_feed_forward
        positionwise_feed_forward = get_positionwise_feed_forward_fn(d_model=d_model,
                                                                     d_inner=d_inner,
                                                                     dropout=dropout,
                                                                     kernel_initializer=initializer,
                                                                     is_training=is_training)

        # 从输入开始构建Transformer-xl结构
        with tf.variable_scope(scope):
            if mems is None:
                mems = [None] * n_layer

            embeddings, shared_params = lookup_fn(inp)
            output = tf.layers.dropout(embeddings, dropout, training=is_training)

            # 相对位置编码
            pos_emb = positional_embedding(klen, d_model, clamp_len)
            pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

            # 多层 （rel_multihead_attn + positionwise_feed_forward）
            for i in range(n_layer):
                # 每层的记忆单元都要输出，作为下一轮的输入
                new_mems.append(_cache_mem(output, mems[i], mem_len))

                with tf.variable_scope('layer_{}'.format(i)):
                    # 第一层attention输入是embedding
                    # 后面层attention输入是上一层attention的output
                    output = attn_layer_func(index=i, rel_inp=pos_emb, w=output, mem=mems[i])

                    # position wise 指每个位置都会共享同样一个FC层
                    output = positionwise_feed_forward(inp=output)

            # 输出层
            output = tf.layers.dropout(output, dropout, training=is_training)
            logit = logit_output(hidden=output, n_token=n_token, params=shared_params)

        return logit, output, new_mems

    return transformer_xl


def build_transformer_xl(inputs, labels, flags, n_token, initializer, proj_initializer):
    per_core_bsz = flags.batch_size // flags.num_core_per_host

    transformer_xl = get_transformer_xl_fn(n_token=n_token,
                                           n_layer=flags.n_layer,
                                           d_model=flags.d_model,
                                           d_embed=flags.d_embed,
                                           n_head=flags.n_head,
                                           d_head=flags.d_head,
                                           d_inner=flags.d_inner,
                                           dropout=flags.dropout,
                                           dropatt=flags.dropatt,
                                           initializer=initializer,
                                           proj_initializer=proj_initializer,
                                           is_training=True,
                                           same_length=flags.same_length,
                                           clamp_len=flags.clamp_len,
                                           untie_r=flags.untie_r)
    _tower_mems, tower_losses, _tower_new_mems, tower_grads_and_vars = [], [], [], []

    # assign_to_gpu(i, ps_device)
    for i in range(flags.num_core_per_host):
        inp = tf.transpose(inputs[i], [1, 0])
        target = tf.transpose(labels[i], [1, 0])

        with tf.device(assign_to_gpu(i)), tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            mems = [tf.placeholder(tf.float32, [flags.mem_len, per_core_bsz, flags.d_model])
                    for _ in range(flags.n_layer)]

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
