# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    loss.py
   Description :
   Author :       Wings DH
   Time：         2020/1/17 1:57 下午
-------------------------------------------------
   Change Activity:
                   2020/1/17: Create
-------------------------------------------------
"""
import tensorflow as tf


def mask_adaptive_logsoftmax(logit_output, target, return_mean=True):
    nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                         logits=logit_output)

    if return_mean:
        nll = tf.reduce_mean(nll)
    return nll


def average_grads_and_vars(tower_grads_and_vars):
    def average_dense(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        grad = grad_and_vars[0][0]
        for g, _ in grad_and_vars[1:]:
            grad += g
        return grad / len(grad_and_vars)

    def average_sparse(grad_and_vars):
        if len(grad_and_vars) == 1:
            return grad_and_vars[0][0]

        indices = []
        values = []
        for g, _ in grad_and_vars:
            indices += [g.indices]
            values += [g.values]
        indices = tf.concat(indices, 0)
        values = tf.concat(values, 0) / len(grad_and_vars)
        return tf.IndexedSlices(values, indices, grad_and_vars[0][0].dense_shape)

    average_grads_and_vars = []
    for grad_and_vars in zip(*tower_grads_and_vars):
        if grad_and_vars[0][0] is None:
            grad = None
        elif isinstance(grad_and_vars[0][0], tf.IndexedSlices):
            grad = average_sparse(grad_and_vars)
        else:
            grad = average_dense(grad_and_vars)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads_and_vars.append(grad_and_var)
    return average_grads_and_vars
