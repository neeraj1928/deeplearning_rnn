import os
import time
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from source.config.constants import *
from source.training.abc_training import ABCTraining


class Config():
    summary_dir = SUMM_AUTO_ENC
    dropout = 0.4  # keep probability = 0.6
    batch_size = BATCH_SIZE
    # number of neurons in hidden cell
    num_hidden_1 = 256
    num_hidden_2 = 128
    # this thing can be evaluated using
    # size of stored embedding matrix
    embedding_size = EMBEDDING_SIZE
    # this can also be detected using
    # embedding matrix
    vocab_size = VOCABULARY_SIZE

    # max sequence length
    max_sequence_length = MAX_ADDR_LENGTH + 1
    adam_lr = 0.001
    sgd_lr = 0.0001
    loss_iteration = 500
    # it will be called 6 times if we have batch size of
    # 10000 sentences and total sentences = 60 million
    validation_iteration = 1000


class LSTMAutoEncoder(ABCTraining):
    """

    """
    def __init__(self, word_tokens):
        super().__init__()
        self.word_tokens = word_tokens

    def get_words_to_indexes(self, words):
        indexes = [self.word_tokens.tokens.get(
            word, self.word_tokens.tokens[UNK]
        ) for word in words]
        return indexes

    def get_batches(self):
        t_input = np.zeros(
            shape=(self.config.batch_size,
                   self.config.max_sequence_length),
            dtype=np.int32)
        t_ip_len = np.zeros(
            shape=(self.config.batch_size), dtype=np.int32)
        num_sentences = 0
        for sentences in self.word_tokens.read_sentences():
            for sentence in sentences:
                sentence.append(0, self.word_tokens.tokens[SOS])
                t_ip_len[num_sentences] = min(len(sentence),
                                              self.config.max_sequence_length)
                if len(sentence) < self.config.max_sequence_length:
                    sentence.extend([EOS for _ in range(
                        self.config.max_sequence_length - len(sentence))])
                elif len(sentence) == self.config.max_sequence_length:
                    sentence[-1] = EOS
                else:
                    sentence = sentence[:self.config.max_sequence_length]
                    sentence[-1] = EOS
                sent_indexes = self.get_words_to_indexes(sentence)
                t_input[num_sentences, :] = sent_indexes
                num_sentences += 1
                if num_sentences == self.config.batch_size:
                    yield (t_input, t_ip_len)
                    num_sentences = 0
                    t_input = np.zeros(
                        shape=(self.config.batch_size,
                               self.config.max_sequence_length),
                        dtype=np.int32)
                    t_ip_len = np.zeros(
                        shape=(self.config.batch_size), dtype=np.int32)
        yield (t_input, t_ip_len)

    def placeholder(self):
        with self.train_graph.as_default():
            with tf.variable_scope("placeholders"):
                with tf.variable_scope("dropout"):
                    self.dropout = tf.placeholder(tf.float32, name="dropout")
                with tf.variable_scope("encoder"):
                    self.enc_inputs = []
                    for i in range(self.config.max_sequence_length):
                        self.enc_inputs.append(
                            tf.placeholder(
                                tf.int32, [None],
                                name="enc_input_{}".format(i)))
                    # to contain the sentences length in each batch
                    self.enc_inp_lengths = tf.placeholder(
                        tf.int32, [None], name="enc_inp_lengths")
                with tf.variable_scope("decoder"):
                    self.dec_inputs = []
                    self.dec_train_labels = []
                    self.dec_label_masks = []
                    for i in range(self.config.max_sequence_length):
                        self.dec_inputs.append(
                            tf.placeholder(
                                tf.int32, [None],
                                name="dec_inputs_{}".format(i)))
                        self.dec_train_labels.append(
                            tf.placeholder(
                                tf.int32, [None],
                                name="dec_inputs_{}".format(i)))
                        self.dec_label_masks.append(
                            tf.placeholder(
                                tf.float32, [None],
                                name="dec_inputs_{}".format(i)))

    def create_feed_dict(self, t_input, t_ip_len, is_train=False):
        feed_dict = {}
        feed_dict[self.enc_inp_lengths] = t_ip_len
        feed_dict[self.dropout] = self.config.dropout if is_train else 0
        for index in range(self.config.max_sequence_length):
            feed_dict[self.enc_inputs[index]] = t_input[:, index]
            feed_dict[self.dec_inputs[index]] = t_input[:, index]
            if index == self.config.max_sequence_length - 1:
                feed_dict[self.dec_train_labels[index]] = t_input[:, index]
            else:
                feed_dict[self.dec_train_labels[index]] = t_input[:, index+1]
            feed_dict[self.dec_label_masks[index]] = \
                ((np.zeros(shape=(self.config.batch_size), dtype=np.int32)
                 + index) < t_ip_len).astype(np.int32)

        return feed_dict

    def embedding(self):
        with self.train_graph.as_default():
            with tf.device('/cpu:0'):
                with tf.variable_scope("embedding"):
                    self.embedding_t = tf.Variable(
                        self.learned_embedding)
                    self.variable_summaries(self.embedding_t)
                    with tf.variable_scope("encoder"):
                        enc_emb = [
                            tf.nn.embedding_lookup(
                                self.embedding_t, batch_words)
                            for batch_words in self.enc_inputs]
                        self.enc_emb = tf.stack(enc_emb, axis=0)
                    with tf.variable_scope("decoder"):
                        dec_emb = [
                            tf.nn.embedding_lookup(self.embedding_t, word)
                            for word in self.dec_inputs]
                        self.dec_emb = tf.stack(dec_emb, axis=0)

    def add_encoder(self):
        """

        """
        with self.train_graph.as_default():
            with tf.variable_scope("encoder"):
                e_cell = tf.contrib.rnn.BasicLSTMCell(
                    self.config.num_hidden_1)
                self.e_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=e_cell, input_keep_prob=(1 - self.dropout))
                self.initial_state = self.e_cell.zero_state(
                    self.config.batch_size, dtype=tf.float32)
                self.e_outputs, self.e_state = tf.nn.dynamic_rnn(
                    self.e_cell, self.enc_emb,
                    initial_state=self.initial_state,
                    sequence_length=source_sequence_length,
                    time_major=True, swap_memory=True)

    def projection_layer(self):
        with self.train_graph.as_default():
            with tf.variable_scope("projection_layer"):
                self.projections = layers_core.Dense(
                    self.config.vocab_size, use_bias=True)

    def add_decoder(self):
        with self.train_graph.as_default():
            with tf.variable_scope("decoder"):
                d_cell = tf.contrib.rnn.BasicLSTMCell(
                    self.config.num_hidden_1)
                self.d_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=d_cell, input_keep_prob=(1 - self.dropout))
                # helper
                self.helper = tf.contrib.seq2seq.TrainingHelper(
                    self.dec_emb,
                    [self.config.max_sequence_length - 1 _ in
                     range(self.config.batch_size)], time_major=True)
                # decoder
                self.decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.d_cell, self.helper, self.e_state,
                    output_layer=self.projections)
                # dynamic decoding
                self.outputs, self.final_context_state, _ = \
                    tf.contrib.seq2seq.dynamic_decode(
                        self.decoder, output_time_major=True,
                        swap_memory=True)

    def model(self):
        self.add_encoder()
        self.projection_layer()
        self.add_decoder()

    def calc_loss(self):
        with self.train_graph.as_default():
            with tf.variable_scope("loss"):
                self.logits = self.outputs.rnn_output
                self.loss = \
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.dec_train_labels, logits=self.logits)

    def calc_cost(self):
        with self.train_graph.as_default():
            with tf.variable_scope("cost"):
                self.cost = (
                    tf.reduce_sum(self.loss * tf.stack(self.dec_label_masks)) /
                    (self.config.batch_size * self.config.max_sequence_length))

    def optimization(self):
        with self.train_graph.as_default():
            with tf.variable_scope('Adam'):
                self.adam_optimizer = tf.train.AdamOptimizer(
                    self.config.adam_lr)

                adam_gradients, v = zip(
                    *self.adam_optimizer.compute_gradients(self.cost))
                adam_gradients, _ = tf.clip_by_global_norm(
                    adam_gradients, 25.0)
                self.adam_optimize = self.adam_optimizer.apply_gradients(
                    zip(adam_gradients, v))

            with tf.variable_scope('SGD'):
                self.sgd_optimizer = tf.train.GradientDescentOptimizer(
                    self.config.sgd_lr)

                sgd_gradients, v = zip(
                    *self.sgd_optimizer.compute_gradients(self.cost))
                sgd_gradients, _ = tf.clip_by_global_norm(
                    sgd_gradients, 25.0)
                self.sgd_optimize = self.sgd_optimizer.apply_gradients(
                    zip(sgd_gradients, v))

    def batch_norm(self):
        pass

    def validate(self, labels, pred):
        rand_idx = np.random.randint(low=1, high=self.config.batch_size)
        act_str, pred_str = "", ""
        for word_index in np.concatenate(
                labels, axis=0)[rand_idx::batch_size].tolist():
            act_str += self.word_tokens.revtokens[word_index] + " "
        for word_index in np.concatenate(
                pred, axis=0)[rand_idx::batch_size].tolist():
            pred_str += self.word_tokens.revtokens[word_index] + " "

        print("Actual: {}\nPredicted: {}".format(act_str, pred_str),
              flush=True)

    def train_model(self):
        with tf.Session(graph=self.train_graph) as sess:
            iteration = 0
            cost = 0
            sess.run(tf.global_variables_initializer())
            for global_step in range(1, self.config.epochs+1):
                optimizer = self.adam_optimize if global_step < 1 else \
                    self.sgd_optimize
                batches = self.get_batches()
                start = time.time()
                for t_input, t_ip_len in batches:
                    feed = self.create_feed_dict(
                        t_input=t_input, t_ip_len=t_ip_len, True)
                    _, t_cost, t_pred = sess.run(
                        [optimizer, self.cost, self.outputs.sample_id],
                        feed_dict=feed)
                    cost += t_cost
                    if iteration % self.config.loss_iteration == 0:
                        end = time.time()
                        print("Epoch {}/{}".format(
                            global_step, self.config.epochs),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(
                                  cost),
                              "{:.4f} sec/loss_iteration".format(
                                  (end-start)))
                        cost = 0
                        start = time.time()
                    if iteration % self.config.validation_iteration == 0:
                        self.validate(feed[self.dec_train_labels], t_pred)
                        self.saver.save(
                            sess, self.config.checkpoint_dir,
                            int("{}{}".format(global_step, iteration)))
                    iteration += 1
                self.saver.save(
                    sess, self.config.checkpoint_dir,
                    int("{}{}".format(global_step, iteration)))

    def run_model(self):
        self.placeholder()
        self.embedding()
        self.model()
        self.calc_loss()
        self.calc_cost()
        self.optimization()
        self.validate()
        self.saver()
        self.train_model()
