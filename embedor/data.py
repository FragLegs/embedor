# -*- coding: utf-8 -*-
import collections
import logging
import math

import numpy as np

log = logging.getLogger(__name__)


class Vocab(object):
    """
    A Vocab object takes in a token stream and creates a one-hot lookup for
    each unique token.

    Parameters
    ----------
    tokens - iterable, optional
        The data to build the vocabulary from. If None, use the `scan()` method
        to build the vocabulary.
    """
    def __init__(self, tokens=None):
        self.counts = collections.Counter()
        self.UNK = 'UNK'
        self.counts[self.UNK] = 0
        self._index = None

        # allow tokens to be passed into the constructor
        if tokens is not None:
            self.scan(tokens)

    def trim(self, min_count):
        """
        Trim the vocbabulary by converting infrequent tokens to self.UNK

        Parameters
        ----------
        min_count - int or float
            How many times a token must appear to keep it.
            If float, the count / total_count of the token must be greater than
            min_count. Must be between 0 and 1.

        Returns
        -------
        self (for composability)
        """
        if min_count < 1 and min_count > 0:
            # get the total count
            total = sum(self.counts.values())

            # get the count that represents this fraction of the total
            min_count = int(math.ceil(min_count * total))

        # for each unique token
        for key, count in self.counts.iteritems():
            # if it is too infrequent
            if count < min_count:
                # get rid of it
                self.counts[key] = 0

                # and add its count to the unknown token
                self.counts[self.UNK] += count

        return self

    def scan(self, tokens):
        """
        Process a token stream to create the one-hot representations.
        Note, this can be called more than once.

        Parameters
        ----------
        tokens - iterable
            The tokens in the data set

        Returns
        -------
        self (for composability)
        """
        if not hasattr(self, 'counts'):
            raise Exception('This Vocab has already been hardened and it '
                            'cannot scan any more data!')

        # get counts of all of the tokens
        new_counts = collections.Counter()
        new_counts[self.UNK] = 0
        new_counts.update(tokens)

        # if we've scanned before
        if len(self.counts) > 1:
            # and we're adding words to the vocabulary
            if set(self.counts.keys()) != set(new_counts.keys()):
                log.warning('Scanning again will change the shape of the '
                            'one-hot vectors.')

        # update the saved counter
        self.counts.update(new_counts)

        return self

    def __getitem__(self, key):
        """
        Return a one-hot vector for this word. If the word isn't in the
        vocabulary, return the one-hot vector for the unknown token.

        Parameters
        ----------
        key - object
            The word to look up

        Returns
        -------
        one-hot numpy array
        """
        # if we don't recognize this word
        if key not in self.index:
            # use the unknown token
            key = self.UNK

        try:
            # if we've precomputed the one hot matrix (trade memory for speed)
            return self.one_hots[self.index[key]]
        except AttributeError:
            # else, construct the one hot vector on the fly
            # initialize the one hot vector to all zeros
            one_hot = np.zeros(len(self), dtype=np.int)

            # set the index of the word to 1
            one_hot[self.index[key]] = 1

            return one_hot

    @property
    def index(self):
        """
        A dictionary of key to position in a list of all keys in the vocab.
        Builds the index lazily on first access.

        Returns
        -------
        dict
        """
        try:
            return self._index
        except AttributeError:
            self._index = dict()
            for k in self.counts.keys():
                self._index[k] = len(self._index)
            return self._index

    def __len__(self):
        """The number of words in the vocabulary (including UNK)."""
        try:
            return len(self.counts)
        except AttributeError:
            return len(self.index)

    def harden(self, precompute=False):
        """
        Get rid of counts, create index, and optionally precomputes
        the one-hot matrix.

        Parameters
        ----------
        precompute - bool, optional
            Whether to precompute the one hot matrix. This can potentially
            consume a LOT of memory.
            Default: False

        Returns
        -------
        self (for composability)
        """
        # create index
        self.index

        # drop counts
        del self.counts

        if precompute:
            # precompute the one hot matrix
            self.one_hots = np.eye(len(self), dtype=np.int)

        return self
