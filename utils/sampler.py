# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(i2i_pairs, i2i_dict, itemnum, batch_size, result_queue):
    def sample():

        pair_ind = np.random.randint(0, len(i2i_pairs[0]))
        s_item, d_item = i2i_pairs[0][pair_ind], i2i_pairs[1][pair_ind]

        interacted_items = i2i_dict[s_item]

        neg = random_neq(0, itemnum, interacted_items)

        return (s_item, d_item, neg)

    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, i2i_pairs, i2i_dict, itemnum, batch_size=64, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(i2i_pairs,
                                                      i2i_dict,
                                                      itemnum,
                                                      batch_size,
                                                      self.result_queue,
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
