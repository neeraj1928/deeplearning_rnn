import os
import time
import random

import numpy as np
import tensorflow as tf

from source.config.constants import *


class Config():
    vocabulary_size = VOCABULARY_SIZE
    embedding_size = EMBEDDING_SIZE
    negative_samples = NUM_NEGATIVE_SAMPLES
    epochs = 15
    batch_size = BATCH_SIZE
    window_size = SKIP_WINDOW_SIZE
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100
    dropout = 0.9
    checkpoint_dir = CKP_WORD2VEC_DIR
    lr = 0.5
    loss_iteration = LOSS_ITERATION
    word_validation_iteration = WORD_VALIDATION_ITERATION


class Word2Vec():
    """

    """
    def __init__(self, word_tokens):
        """
        @param word_tokens: this is class CreateWordTokens
        """
        self.train_graph = tf.Graph()
        self.config = Config()
        self.word_tokens = word_tokens
        self.run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    def get_words_to_indexes(self, words):
        indexes = []
        [indexes.append(self.word_tokens.tokens.get(
            word, self.word_tokens.tokens[UNK]
        ))]
        # for word in words:
        #     if np.floor(np.log10(
        #             self.word_tokens.tokenfreq.get(word, 1))) > 2:
        #         index = self.word_tokens.tokens.get(
        #             word, self.word_tokens.tokens[UNK])
        #     else:
        #         index = self.word_tokens.tokens[UNK]
        #     indexes.append(index)
        return indexes

    def get_target(self, address, idx):
        ''' Get a list of words in a window around an index. '''

        R = np.random.randint(1, self.config.window_size+1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = set(address[start:idx] + address[idx+1:stop+1])
        indexes = self.get_words_to_indexes(target_words)

        return indexes

    def get_batches(self):
        '''
            Create a generator of word batches as a tuple (inputs, targets)
        '''
        num_words = 0
        for sentences in self.word_tokens.read_sentences():
            x = np.zeros(shape=(self.config.batch_size), dtype=np.int32)
            y = np.zeros(shape=(self.config.batch_size, 1), dtype=np.int32)
            for sentence in sentences:
                for i in range(len(sentence)):
                    batch_x = self.get_words_to_indexes([sentence[i]])[0]
                    batch_y = self.get_target(sentence, i)
                    if num_words + len(batch_y) < self.config.batch_size:
                        x[num_words: num_words + len(batch_y)] = \
                            [batch_x] * len(batch_y)
                        y[num_words: num_words + len(batch_y), 0] = batch_y
                    else:
                        if num_words + len(batch_y) == \
                                self.config.batch_size:
                            x[num_words: num_words + len(batch_y)] = \
                                [batch_x] * len(batch_y)
                            y[num_words: num_words + len(batch_y), 0] = batch_y
                        else:
                            remaining = self.config.batch_size - num_words
                            x[num_words: num_words + remaining] = \
                                [batch_x] * remaining
                            y[num_words: num_words + remaining, 0] = \
                                batch_y[:remaining]
                        yield x, y
                        x = np.zeros(
                            shape=(self.config.batch_size), dtype=np.int32)
                        y = np.zeros(
                            shape=(self.config.batch_size, 1), dtype=np.int32)
                        num_words = 0
                        continue
                    num_words += len(batch_y)

    def placeholder(self):
        with self.train_graph.as_default():
            self.inputs = tf.placeholder(tf.int32, [None], name='inputs')
            self.labels = tf.placeholder(tf.int32, [None, None], name='labels')
            self.dropout = tf.placeholder(tf.int32, name='dropout')

    def create_feed_dict(self, input_batch, dropout, label_batch=None):
        feed_dict = {
            self.inputs: input_batch,
            self.dropout: dropout
        }
        if label_batch is not None:
            feed_dict[self.labels] = label_batch

        return feed_dict

    def embedding(self):
        with self.train_graph.as_default():
            with tf.device('/cpu:0'):
                self.embedding = tf.Variable(
                    tf.random_uniform(
                        (self.config.vocabulary_size,
                         self.config.embedding_size), -1, 1))
                # use tf.nn.embedding_lookup to get the hidden layer output
                self.embed = tf.nn.embedding_lookup(
                    self.embedding, self.inputs)

    def batch_norm(self):
        with self.train_graph.as_default():
            norm = tf.sqrt(
                tf.reduce_sum(tf.square(self.embedding), 1, keepdims=True))
            self.embedding = self.embedding / norm

    def model(self):
        with self.train_graph.as_default():
            # create softmax weight matrix here
            self.softmax_w = tf.Variable(
                tf.truncated_normal(
                    (self.config.vocabulary_size, self.config.embedding_size)))
            # create softmax biases here
            self.softmax_b = tf.Variable(
                tf.zeros(self.config.vocabulary_size), name="softmax_bias")

    def calc_loss(self):
        # Calculate the loss using negative sampling
        with self.train_graph.as_default():
            self.loss = tf.nn.sampled_softmax_loss(
                weights=self.softmax_w,
                biases=self.softmax_b,
                labels=self.labels,
                inputs=self.embed,
                num_sampled=self.config.negative_samples,
                num_classes=self.config.vocabulary_size,
                partition_strategy="div")

    def calc_cost(self):
        with self.train_graph.as_default():
            self.cost = tf.reduce_mean(self.loss)

    def optimization(self):
        with self.train_graph.as_default():
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

    def validate(self):
        with self.train_graph.as_default():
            # TODO: understant below implementation
            # # From Thushan Ganegedara's implementation
            # not clear of sampling thing
            # pick 8 samples from (0,100) and (1000,1100) each ranges.
            # lower id implies more frequent
            # import pdb; pdb.set_trace()
            self.valid_examples = np.array(
                random.sample(range(self.config.valid_window),
                              self.config.valid_size//2))
            self.valid_examples = np.append(
                self.valid_examples,
                random.sample(range(1000, 1000+self.config.valid_window),
                              self.config.valid_size//2))

            self.valid_dataset = tf.constant(
                self.valid_examples, dtype=tf.int32)

            # We use the cosine distance:
            valid_embedding = tf.nn.embedding_lookup(
                self.embedding, self.valid_dataset)
            self.similarity = tf.matmul(
                valid_embedding, tf.transpose(self.embedding))

    def saver(self):
        with self.train_graph.as_default():
            self.saver = tf.train.Saver()

    def train_model(self):
        with tf.Session(graph=self.train_graph) as sess:
            iteration = 0
            loss = 0
            sess.run(tf.global_variables_initializer())
            for global_step in range(1, self.config.epochs+1):
                batches = self.get_batches()
                start = time.time()
                for x, y in batches:
                    feed = self.create_feed_dict(
                        input_batch=x,
                        dropout=self.config.dropout, label_batch=y)
                    train_loss, _ = sess.run(
                        [self.cost, self.optimizer], feed_dict=feed,
                        options=self.run_opts)

                    loss += train_loss

                    if iteration % self.config.loss_iteration == 0:
                        end = time.time()
                        print("Epoch {}/{}".format(
                            global_step, self.config.epochs),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(
                                  loss/self.config.loss_iteration),
                              "{:.4f} sec/loss_iteration".format(
                                  (end-start)/self.config.loss_iteration))
                        loss = 0
                        start = time.time()

                    if iteration % self.config.word_validation_iteration == 0:
                        # TODO: not very cleasr
                        # note that this is expensive
                        sim = self.similarity.eval()
                        for i in range(self.config.valid_size):
                            valid_word = self.word_tokens.revtokens[
                                self.valid_examples[i]]
                            top_k = 8  # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k+1]
                            log = 'Nearest to {}:'.format(valid_word)
                            for k in range(top_k):
                                close_word = self.word_tokens.revtokens[
                                    nearest[k]]
                                log = '{} {},'.format(log, close_word)
                            print(log)
                    iteration += 1
                # sess.run(self.batch_norm())
                # do batch normalization
                norm = tf.sqrt(
                    tf.reduce_sum(tf.square(self.embedding), 1, keepdims=True))
                self.embedding = self.embedding / norm
                save_path = self.saver.save(
                    sess, self.config.checkpoint_dir, global_step)

    def run_model(self):
        # import pdb; pdb.set_trace()
        self.placeholder()
        self.embedding()
        self.model()
        self.calc_loss()
        self.calc_cost()
        self.optimization()
        self.validate()
        self.saver()
        self.train_model()
