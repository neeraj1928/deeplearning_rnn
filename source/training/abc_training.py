import os
import time
import random

import numpy as np
import tensorflow as tf


class Config():
    raise NotImplementedError


class ABCTraining():
    """

    """
    def __init__(self):
        self.train_graph = tf.Graph()
        self.config = Config()

    def get_batches(self):
        raise NotImplementedError

    def placeholder(self):
        raise NotImplementedError

    def create_feed_dict(self):
        raise NotImplementedError

    def embedding(self):
        raise NotImplementedError

    def model(self):
        raise NotImplementedError

    def calc_loss(self):
        raise NotImplementedError

    def calc_cost(self):
        raise NotImplementedError

    def optimization(self):
        raise NotImplementedError

    def batch_norm(self):
        raise NotImplementedError

    def variable_summaries(self, var):
        """
            Attach a lot of summaries to a Tensor
            (for TensorBoard visualization).
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

    def validate(self):
        raise NotImplementedError

    def saver(self):
        with self.train_graph.as_default():
            self.saver = tf.train.Saver()

    def train_model(self):
        raise NotImplementedError

    def run_model(self):
        raise NotImplementedError
