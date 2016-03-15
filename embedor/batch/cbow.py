# -*- coding: utf-8 -*-
import collections
import logging

import numpy as np

import embedor.batch.base

log = logging.getLogger(__name__)


class CBOW(embedor.batch.base.Batches):
    """
    Create training batches for continuous bag of words embeddings, in which
    the context words predict the target word

    Parameters
    ----------
    vocab - Vocab
        The vocabulary to use
    batch_size - int, optional
        How many train/test pairs to generate per batch
        Default: 64
    context - tuple of int, optional
        How many words from the left and right to use as context for the
        target word
        Default: (2, 2)
    """
    def __init__(self, vocab, batch_size=64, tokens=None, context=(2, 2)):
        super(CBOW, self).__init__(vocab=vocab,
                                   batch_size=batch_size,
                                   tokens=tokens)
        self.left, self.right = context

    def next(self):
        # vocab must have already been scanned
        assert(len(self.vocab) > 0)

        # determine the full window size (context + target)
        window_size = self.left + 1 + self.right

        # set up a buffer for tokens
        context = collections.deque(maxlen=window_size)

        # collect batch_size contexts and targets
        data, labels = [], []

        # for each token
        for t in self.tokens:
            # add this token to the deque
            context.append(self.vocab[t])

            # if we've built up enough context
            if len(context) == window_size:
                # add the target and the surrounding context to the batch
                # listify the context
                context_list = list(context)
                labels.append(context_list.pop(self.left))
                data.append(context_list)

                # if we have enough for a batch
                if len(labels) == self.batch_size:
                    # yield the data as numpy arrays
                    return np.array(data), np.array(labels)

                    # reset the batch collector
                    data, labels = [], []
