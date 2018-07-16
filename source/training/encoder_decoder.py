import os
import time
import random

import numpy as np
import tensorflow as tf

from source.config.constants import *


class Config():
    summary_dir = SUMM_AUTO_ENC
    dropout = 0.7
    batch_size = BATCH_SIZE


class EncoderDecoder():
    """

    """

    def __init__(self):
        self.config = Config()
        self.train_graph = tf.Graph()

    def placeholder(self):
        with self.train_graph.as_default():
            with tf.name_scope("placeholders"):
                pass

    def embedding(self):
        with self.train_graph.as_default():
            with tf.device('/cpu:0'):
                with tf.name_scope("embedding"):
                    self.embedding_t = tf.Variable(
                        self.learned_embedding)
                    self.variable_summaries(self.embedding_t)
                    # combination of trainable and learned embedding
                    # can be used
                    self.embedding_c = tf.constant(self.learned_embedding)
                    # initially using trainable embedding only
                    self.embed = tf.nn.embedding_lookup(
                        self.embedding_t, self.inputs)
