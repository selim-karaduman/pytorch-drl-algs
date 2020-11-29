import numpy as np
import random

"""
https://github.com/hill-a/stable-baselines/blob/master/
stable_baselines/common/segment_tree.py
"""

class SegmentTree(object):

    def __init__(self, size, function, identity):
        self.size = size
        self.internal_offset = self.size - 1
        self.segment_tree = np.zeros(size * 2 - 1) + identity
        self.function = function
        
    def _propagate(self, ind):
        if(ind <= 0): return
        parent = (ind - 1) // 2
        left_ind = parent*2 + 1
        right_ind = parent*2 + 2
        self.segment_tree[parent] = self.function(self.segment_tree[left_ind], 
            self.segment_tree[right_ind])
        self._propagate(parent)

    def update(self, ind, priority):
        segtree_ind = self.internal_offset + ind
        self.segment_tree[segtree_ind] = priority
        self._propagate(segtree_ind)

    def __setitem__(self, ind, priority):
        self.update(ind, priority)

    def __getitem__(self, ind):
        return self.segment_tree[self.internal_offset + ind]

#-----------------------------------------------------------------------

class SumTree(SegmentTree):
    def __init__(self, size):
        super().__init__(size, np.add, 0)

    def sample_batch_idx(self, batch_size):
        p_total = self.segment_tree[0]
        step_size = p_total / batch_size
        indices = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            val = random.uniform(step_size*i, step_size*(i+1))
            idx = self._sample_idx(val)
            indices[0] = idx
        return indices

    def get_sum(self):
        return self.segment_tree[0]

    def _sample_idx(self, val, ind=0):
        if(ind >= self.internal_offset): 
            return ind - self.internal_offset
        left_ind = ind*2 + 1
        right_ind = ind*2 + 2
        rval = val - self.segment_tree[left_ind]
        if(val <= self.segment_tree[left_ind]):
            s = self._sample_idx(val, left_ind)
        else:
            s = self._sample_idx(rval, right_ind)
        return s


#-----------------------------------------------------------------------

class MinTree(SegmentTree):

    def __init__(self, size):
        super().__init__(size, min, float("inf"))

    def get_min(self):
        return self.segment_tree[0]
