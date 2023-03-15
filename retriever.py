from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np


class Retreiver():
    def __init__(self, sample_strategy, warm_up):
        self.sample_strategy = sample_strategy
        self.warm_up = warm_up

    def sample(self, pool, expert_pool, size, train_idx):
        if self.sample_strategy == "kmeans":
            sampler = KMeans(
                n_clusters=size
            )
            labels = sampler.fit_predict(pool)
            label2ind = dict()
            for i, label in enumerate(labels):
                if label in label2ind.keys():
                    label2ind[label].append(i)
                else:
                    label2ind[label] = [i]
            ind = [random.sample(idx, 1) for _, idx in label2ind.items()]
        elif self.sample_strategy == "cos_diff":
            sim = cosine_similarity(pool, expert_pool)
            # sim: (pool_size, expert_size) - (2048, 1024)
            sim_mean = np.mean(sim, 1)
            ind = np.argpartition(sim_mean, -size)[:size]
        elif self.sample_strategy == "cos_sim":
            sim = cosine_similarity(pool, expert_pool)
            # sim: (pool_size, expert_size) - (2048, 1024)
            sim_mean = np.mean(sim, 1)
            ind = np.argpartition(sim_mean, -size)[-size:]
        elif self.sample_strategy == "scheduling":
            if train_idx < self.warm_up:
                sim = cosine_similarity(pool, expert_pool)
                # sim: (pool_size, expert_size) - (2048, 1024)
                sim_mean = np.mean(sim, 1)
                ind = np.argpartition(sim_mean, -size)[:size]
            else:
                sim = cosine_similarity(pool, expert_pool)
                # sim: (pool_size, expert_size) - (2048, 1024)
                sim_mean = np.mean(sim, 1)
                ind = np.argpartition(sim_mean, -size)[-size:]
        else:
            raise NotImplementedError
        return ind