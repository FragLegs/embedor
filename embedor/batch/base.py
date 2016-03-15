# -*- coding: utf-8 -*-
import abc
import logging

log = logging.getLogger(__name__)


class Batches(object):
    """
    Batches creates batches of training data and labels from the
    text.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, vocab, batch_size=64, tokens=None):
        """
        Parameters
        ----------
        vocab - Vocab object
            The vocabulary to use when generating batches
        batch_size - int, optional
            The size of each batch
            Default: 64
        tokens - iterable of tokens
            The tokens to create batches from. Set here or manually later.
            Default: None
        """
        self.vocab = vocab
        self.batch_size = batch_size
        self.tokens = tokens

    def __iter__(self):
        """
        Allow user to iterate over batches
        """
        return self

    @abc.abstractmethod
    def next():
        pass
