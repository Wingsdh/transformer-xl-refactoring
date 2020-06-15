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
from abc import ABC
import math
import os
import time

from common.log import logger
import tensorflow as tf


class BaseHook(ABC):

    def begin(self, sess, init_step):
        pass

    def before_run(self):
        pass

    def after_run(self, step, loss, lr):
        pass

    def end(self):
        pass


class TimeLogHook(BaseHook):

    def __init__(self, interval):
        """
        Log every <interval> seconds

        @param interval: int, How long between every log.
        """
        self.__prev_time = time.time()
        self.interval = interval
        self.__total_loss = 0
        self.__prev_step = 0

    def begin(self, sess, init_step):
        logger.info('TimeLog start to hook')
        self.__prev_time = time.time()
        self.__prev_step = init_step

    def after_run(self, step, loss, lr):
        self.__total_loss += loss

        now_interval = time.time() - self.__prev_time
        if now_interval > self.interval:
            curr_loss = self.__total_loss / (step - self.__prev_step)
            logger.info(
                "[{}] lr {:8.6f} | loss {:.2f} | pplx {:>7.2f} | bpc {:>7.4f} ".format(
                    step, lr, curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
            self.__prev_time = time.time()
            self.__total_loss = 0
            self.__prev_step = 0


class StepLogHook(BaseHook):

    def __init__(self, log_steps):
        """

        @param log_steps: `int`, save every N steps.

        """
        self.__prev_step = 0
        self.log_steps = log_steps
        self.__total_loss = 0

    def begin(self, sess, init_step):
        self.__prev_step = init_step

    def after_run(self, step, loss, lr):
        self.__total_loss += loss

        interval_steps = step - self.__prev_step
        if interval_steps > self.log_steps:
            curr_loss = self.__total_loss / (step - self.__prev_step)
            logger.info(
                "[{}] lr {:8.6f} | loss {:.2f} | pplx {:>7.2f} | bpc {:>7.4f} ".format(
                    step, lr, curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
            self.__total_loss = 0
            self.__prev_step = step


class SaverHook(BaseHook):
    def __init__(self, model_d_path, warm_start_d_path=None):
        """
        @param model_d_path: str, path to save model
        @param warm_start_d_path: path to warm start
        """
        self.model_d_path = model_d_path
        if self.warm_start_d_path and os.path.exists(self.warm_start_d_path):
            self.warm_start_d_path = warm_start_d_path
        else:
            self.warm_start_d_path = model_d_path

        if os.path.exists(self.model_d_path):
            os.makedirs(self.model_d_path)

        self._saver = tf.train.Saver()
        self._sess = None

    def begin(self, sess, init_step):
        self._sess = sess
        logger.info("warm start from {}".format(self.warm_start_d_path))
        try:
            init_ckpt_path = tf.train.latest_checkpoint(self.warm_start_d_path)
            self._saver.restore(sess, init_ckpt_path)
        except ValueError:
            logger.warning('restore fail invalid path:{}'.format(self.warm_start_d_path))

    def save_check_point(self, step, loss):
        save_path = os.path.join(self.model_d_path, "model_{}_{.:4}.ckpt".format(step, loss))
        self._saver.restore(self._sess, save_path)
        logger.info("Save Model saved in path: {}".format(save_path))


class TimeSaverHook(SaverHook):
    """
    Judge whether to save by time.
    """

    def __init__(self, interval, model_d_path, warm_start_d_path=None):
        """

        @param interval:
        @param model_d_path: str, path to save model
        @param warm_start_d_path: path to warm start
        """
        super(TimeSaverHook, self).__init__(model_d_path=model_d_path,
                                            warm_start_d_path=warm_start_d_path)
        self.interval = interval
        self.__prev_time = time.time()
        self.__total_loss = 0
        self.__prev_step = 0

    def begin(self, sess, init_step):
        self.__prev_step = init_step

    def after_run(self, step, loss, lr):
        self.__total_loss += loss
        if time.time() - self.__prev_time > self.interval:
            curr_loss = self.__total_loss / (step - self.__prev_step)
            self.save_check_point(step, curr_loss)
            self.__prev_step = step
            self.__total_loss = 0
            self.__prev_time = time.time()


class StepSaverHook(SaverHook):
    """
    Judge whether to save by steps.
    """

    def __init__(self, save_steps, model_d_path, warm_start_d_path=None):
        """
        Save checkpoint every {save_steps}
        @param save_steps: int, every {save_steps} steps trigger a save process
        @param model_d_path: str, path to save model
        @param warm_start_d_path: path to warm start
        """
        super(StepSaverHook, self).__init__(model_d_path=model_d_path,
                                            warm_start_d_path=warm_start_d_path)
        self.save_steps = save_steps
        self.__total_loss = 0
        self.__prev_step = 0

    def begin(self, sess, init_step):
        self.__prev_step = init_step

    def after_run(self, step, loss, lr):
        self.__total_loss += loss
        if step - self.__prev_step > self.save_steps:
            curr_loss = self.__total_loss / (step - self.__prev_step)
            self.save_check_point(step, curr_loss)
            self.__prev_step = step
            self.__total_loss = 0
