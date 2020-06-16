# -*- encoding: utf-8 -*-
"""
-------------------------------------------------
   File Name：    train_hook.py
   Description :
   Author :       Wings DH
   Time：         2020/6/15 1:57 下午
-------------------------------------------------
   Change Activity:
                   2020/6/15: Create
-------------------------------------------------
"""
import math

import numpy as np
from tensorflow.python.training.basic_session_run_hooks import LoggingTensorHook
from tensorflow.python.training.session_run_hook import SessionRunArgs

from common.log import logger


class TransXlTrainLogHook(LoggingTensorHook):
    """
    inherit from LoggingTensorHook
    1. overwrite the method "_log_tensors" to adjust my logger and format
    2. overwrite after_run to log mean loss
    """
    KEY_LOSS = 'loss'
    KEY_LEARN_RATE = 'learn_rate'
    KEY_STEP = 'step'

    @classmethod
    def _format(cls, tensor_values):
        step = tensor_values[cls.KEY_STEP]
        loss = tensor_values[cls.KEY_LOSS]
        lr = tensor_values[cls.KEY_LEARN_RATE]
        return "[{:5}] lr {:8.6f} | loss {:.4f} | pplx {:>7.2f} | bpc {:>7.4f}".format(
            step, lr, loss, math.exp(loss), loss / math.log(2))

    def __init__(self, loss_tensor, learn_rate_tensor, step_tensor, *args, **kwargs):

        tensors = {self.KEY_LOSS: loss_tensor,
                   self.KEY_LEARN_RATE: learn_rate_tensor,
                   self.KEY_STEP: step_tensor}

        super(TransXlTrainLogHook, self).__init__(tensors=tensors,
                                                  formatter=self._format,
                                                  *args, **kwargs)
        self._total_loss = 0
        self._n_step = 0
        self._should_trigger = None

    def before_run(self, run_context):
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        # Run every iter so that can calculate a mean value
        return SessionRunArgs(self._current_tensors)

    def after_run(self, run_context, run_values):
        """
        calculate mean loss and log
        @param run_context:
        @param run_values:
        @return:
        """
        _ = run_context
        loss = run_values.results[self.KEY_LOSS]
        self._total_loss += loss
        self._n_step += 1
        if self._should_trigger and self._n_step > 0:
            run_values.results[self.KEY_LOSS] = self._total_loss / self._n_step
            self._log_tensors(run_values.results)
            self._total_loss = 0
            self._n_step = 0

        self._iter_count += 1

    def _log_tensors(self, tensor_values):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
        if self._formatter:
            logger.info(self._formatter(tensor_values))
        else:
            stats = []
            for tag in self._tag_order:
                stats.append("{} = {}".format(tag, tensor_values[tag]))
            if elapsed_secs is not None:
                logger.info("{} ({:.3} sec)".format(", ".join(stats), elapsed_secs))
            else:
                logger.info("{}".format(", ".join(stats)))
        np.set_printoptions(**original)

