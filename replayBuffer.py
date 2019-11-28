import collections
import numpy as np


class ReplayBuffer:

    def __init__(self, set_batch_size):
        self.transition_container = collections.deque(maxlen=5000)
        self.weight_buffer = collections.deque(maxlen=5000)
        self._batch_size = set_batch_size
        self.transition_probabilities = None
        self.indexes = None

    def random_sampling(self, replace):
        container_size = len(self.transition_container)

        range_size = range(container_size)
        if replace:
            indexes = np.random.choice(range_size, self._batch_size, replace=True)
            data_set = np.array(self.transition_container)
            self.indexes = indexes
            return data_set[indexes]
        else:
            indexes = np.random.choice(range_size, self._batch_size,
                                       p=self.transition_probabilities, replace=False)

            data_set = np.array(self.transition_container)
            self.indexes = indexes
            return data_set[indexes]

    def add(self, transition):
        self.transition_container.append(transition)

        if len(self.weight_buffer) == 0:
            self.weight_buffer.append(1)
        else:
            self.weight_buffer.append(max(self.weight_buffer))

        self.transition_probabilities = np.array(self.weight_buffer) / sum(self.weight_buffer)

    def update_weight_buffer(self, delta):
        for delta_index, index in enumerate(self.indexes):
            self.weight_buffer[index] = delta[delta_index]
