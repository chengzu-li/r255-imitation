import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import seals  # needed to load environments
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
import numpy as np
import random
import torch
import os

from customize_model import CustomizeTrainer

from sb3_contrib import TRPO

seed = 0

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

simulation_env = "seals/Ant-v0"
# TODO:
#  ["seals/CartPole-v0", "seals/MountainCar-v0", "seals/Ant-v0", "seals/HalfCheetah-v0",
#  "seals/Hopper-v0", "seals/Humanoid-v0", "seals/Humanoid-v0", "seals/Humanoid-v0"]

env = gym.make(simulation_env)


expert = PPO(
    policy=FeedForward32Policy(features_extractor_class=NormalizeFeaturesExtractor),
    env=env,
    seed=seed,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
)

expert.learn(100000)  # Note: set to 100000 to train a proficient expert

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    make_vec_env(
        simulation_env,
        n_envs=5,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    ),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)

venv = make_vec_env(simulation_env, n_envs=8, rng=rng)
learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    # n_steps=2048        # change the steps
)
# learner = TRPO(
#     policy="MlpPolicy",
#     env=venv,
#     batch_size=64,
#     learning_rate=0.0003,
#     seed=10
# )

reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)
gail_trainer = CustomizeTrainer(
    sample_strategy="random",       # TODO: random, kmeans, cos_diff, cos_sim,
    sample_key="obs",               # TODO: obs, acts, next_obs, dones, infos
    demonstrations=rollouts,
    demo_batch_size=1024,           # FIXME: 改变batch_size, 1024
    gen_replay_buffer_capacity=2048,        # FIXME: 这个会影响sample pool的大小，可以把它改成None就存下所有的samples了
    n_disc_updates_per_round=4,         # FIXME: The number of discriminator updates after each round of generator updates in AdversarialTrainer.learn().
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)
gail_trainer.train(300000)  # Note: set to 300000 for better results
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')

print(np.mean(learner_rewards_after_training))
print(np.mean(learner_rewards_before_training))

plt.hist(
    [np.array(learner_rewards_before_training), np.array(learner_rewards_after_training)],
    label=["untrained", "trained"],
)
plt.legend()
plt.show()