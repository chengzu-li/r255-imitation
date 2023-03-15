import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

import os
import pickle

import numpy as np

NAME_MAPPING = {
    "cos_diff": "CosineDiff",
    "cos_sim": "CosineSim",
    "scheduling": "CosineSchedule"
}

dataset_name = "Swimmer-v0"
sampling = "cos_diff"
all_sampling = ['cos_diff', 'cos_sim', 'scheduling']
all_dataset_name = ['MountainCar', 'HalfCheetah', 'Swimmer', 'Reacher']
sample_size = 512
pool_size = 2048

loss_record = []
t = 0
plt.figure(figsize=(5*len(all_dataset_name), 5), dpi=80)
for i, dataset_name in enumerate(all_dataset_name):
    loss_record = []
    for sampling in all_sampling:
        if dataset_name == "Reacher":
            dir_path = f"outputs/{dataset_name}-v2/ppo-mlp_policy/{sampling}-sample_{sample_size}_from_{pool_size}-repeat3"
        else:
            dir_path = f"outputs/seals/{dataset_name}-v0/ppo-mlp_policy/{sampling}-sample_{sample_size}_from_{pool_size}-repeat3"
        file_path = os.path.join(dir_path, 'loss.pickle')

        with open(file_path, 'rb') as f:
            loss = pickle.load(f)

        loss = np.mean(np.array(loss), 0)
        loss_record.append(loss)
    plt.subplot(int(f"1{len(all_dataset_name)}{i+1}"))
    plt.title(dataset_name)
    x = np.array([i for i in range(len(loss_record[0]))])
    for i, sampling in enumerate(all_sampling):
        plt.plot(x, loss_record[i])
    plt.legend(all_sampling, loc="upper right")
plt.savefig("loss.pdf", format='pdf', bbox_inches='tight')
plt.show()