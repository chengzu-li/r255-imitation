from imitation.data.buffer import ReplayBuffer
from imitation.data import types

import numpy as np

from retriever import Retreiver


class CustomizeReplayBuffer(ReplayBuffer):
    def __init__(self, sample_strategy, sample_key, warm_up, **kwargs):
        super().__init__(**kwargs)
        self.sample_strategy = sample_strategy
        self.sample_key = sample_key

        self.retriever = Retreiver(sample_strategy=sample_strategy, warm_up=warm_up)

    def sample(self, n_samples: int, expert_samples: dict, idx: int) -> types.Transitions:
        """Uniformly sample `n_samples` samples from the buffer with replacement.

                Args:
                    n_samples: The number of samples to randomly sample.
                    expert_samples: expert samples

                Returns:
                    samples (np.ndarray): An array with shape
                        `(n_samples) + self.sample_shape`.

                Raises:
                    ValueError: The buffer is empty.
                """
        if self.size() == 0:
            raise ValueError("Buffer is empty")
        buffer_dict = self._buffer

        if self.sample_strategy == "random":
            ind = np.random.randint(self.size(), size=n_samples)
        else:
            expert_pool = expert_samples[self.sample_key]
            pool = buffer_dict._arrays[self.sample_key]
            ind = self.retriever.sample(pool=pool, expert_pool=expert_pool, size=n_samples, train_idx=idx)

        retrieved_samples = {k: buffer[ind].squeeze() for k, buffer in buffer_dict._arrays.items()}
        return types.Transitions(**retrieved_samples)