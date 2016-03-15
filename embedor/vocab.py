# -*- coding: utf-8 -*-
import collections
import logging

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
    min_count - int, optional
        How many times we need to see a token before we add it to the vocab
        Default: 1
    """
    def __init__(self, tokens=None, min_count=1):
        self.counts = collections.Counter()

        self.UNK = 'UNK'
        self.word2index = {self.UNK: 0}
        self.words = [self.UNK]

        assert(min_count > 0 and isinstance(min_count, int))
        self.min_count = min_count

        # track how many total tokens we've seen
        self.total_seen = 0

        # allow tokens to be passed into the constructor
        if tokens is not None:
            self.scan(tokens)

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
        if len(self) > 1:
            log.warning('Scanning a second time may add to the total number of'
                        ' words in the vocab. Be careful when indexing one-hot'
                        ' vectors.')

        for t in tokens:
            # count the token
            self.counts[t] += 1

            # update total seen
            self.total_seen += 1

            # when we pass the minimum count threshold, add it to the voab
            if self.counts[t] == self.min_count:
                self.word2index[t] = len(self.words)
                self.words.append(t)

        return self

    def __getitem__(self, key):
        """
        Return the index for this word. If the word isn't in the
        vocabulary, return the index for the unknown token.

        Parameters
        ----------
        key - object
            The word to look up

        Returns
        -------
        int
        """
        return self.word2index.get(key, self.word2index[self.UNK])

    def __len__(self):
        """The number of words in the vocabulary (including UNK)."""
        return len(self.words)

    def probability(self, key):
        """
        Get the empirical probability of a word

        Parameters
        ----------
        key - object
            The word to look up

        Returns
        -------
        float
        """
        return float(self.count[key]) / self.total_seen
