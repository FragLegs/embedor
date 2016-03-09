# -*- coding: utf-8 -*-
import logging

import tensorflow as tf

log = logging.getLogger(__name__)


class Embedding(object):
    """
    Use this class to create embedding vectors from context and labels.

    Parameters
    ----------
    vocab_size - int
        How many unique tokens exist in the vocabulary
    batch_size - int
        How many tokens per training batch
    embedding_size - int, optional
        The size of the embedding vector
        Default: 50
    """
    def __init__(self, vocab_size, batch_size, embedding_size=50):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size

    def build_net(self):
        """
        Constructs the network that does the embedding

        Returns
        -------
        self
        """
        # set up the neural net graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            # add placeholders for the data
            self.train_data = tf.placeholder(tf.int32,
                                             shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32,
                                               shape=[self.batch_size, 1])

            # the embeddings
            # This is a [vocab, embedding size] matrix where each row
            # represents a unique token in the vocabulary.
            # The network will look up
            self.embeddings = tf.Variable(tf.random_uniform(
                [self.vocab_size, self.embedding_size], -1.0, 1.0))

            # the output layer (unused except in training the embedding layer)
            self.hidden_weights = tf.Variable(tf.truncated_normal(
                [self.vocab_size, self.embedding_size]))