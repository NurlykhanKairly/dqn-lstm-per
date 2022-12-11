# Source 1: https://github.com/rlcode/per/blob/master/prioritized_memory.py
# Source 2: https://github.com/rlcode/per/blob/master/SumTree.py

import random
import numpy as np


class Memory:
    def __init__(self, capacity):
        self.beta = 0.3
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def add(self, error, sample):
        self.tree.add((np.abs(error) + 0.05) ** 0.85, sample)

    def sampling_weight(self, priorities):
        sampling_probabilities = priorities / self.tree.total()
        s_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        return s_weight / s_weight.max()

    def sample(self, n):
        self.beta = np.min([1., self.beta + 0.0005])

        segment = self.tree.total() / n
        batch, priorities = [], []
        for i in range(n):
            s = random.uniform(segment * i, segment * (i + 1))
            (_, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)

        return batch, self.sampling_weight(priorities)

    def update(self, idx, error):
        self.tree.update(idx, (np.abs(error) + 0.05) ** 0.85)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s > self.tree[left]:
            return self._retrieve(right, s - self.tree[left])
        else:
            return self._retrieve(left, s)

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
