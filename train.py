import gym
import pickle
import imitation
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
import numpy as np
import random
import torch
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')

from customize_model import CustomizeTrainer
from utils import get_expert_args
from sb3_contrib import TRPO

from collections import OrderedDict


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def init_agent(po_algo: str, env: gym.Env, seed: int, **kwargs):
    # if policy == "mlp":
    #     expert_policy = MlpPolicy
    # else:
    #     raise NotImplementedError

    if po_algo == "ppo":
        expert = PPO(
            env=env,
            seed=seed,
            **kwargs
        )
    elif po_algo == "trpo":
        expert = TRPO(
            env=env,
            seed=seed,
            **kwargs
        )
    else:
        raise NotImplementedError

    return expert

# def get_learner_args(po_algo: str, task_name: str):
#     task_name = task_name.split("/")[-1]
#     with open(f"learner_args/{task_name}/{po_algo}.json", "r", encoding='utf-8') as f:
#         agent_args = json.load(f)
#     return agent_args


def train(
        seed: int,
        simulation_env: str,
        policy: str,
        po_algo: str,
        sample_strategy: str,
        demo_batch_size: int,
        sample_pool_size: int,
        n_disc_updates_per_round: int,
        expert_learn_step: int,
        learner_learn_step: int,
        warm_up: int
):
    '''

    :param seed:
    :param simulation_env:
    ################## Experiment Variables #########################
    :param policy:
    :param po_algo:
    :param sample_strategy: sample strategy
    :param demo_batch_size: control the batch size
    :param sample_pool_size: control the selection pool size, if set to None, then retrieve from all samples
    #################################################################
    :param n_disc_updates_per_round: The number of discriminator updates after each round of generator updates in
                                     AdversarialTrainer.learn().
    :param expert_learn_step:
    :param learner_learn_step:
    :return:
    '''
    set_seed(seed)
    env = gym.make(simulation_env)

    expert_args = get_expert_args(po_algo, task_name=simulation_env)

    if "n_envs" in expert_args.keys():
        n_envs = expert_args.pop('n_envs')
    else:
        n_envs = 5
    n_timesteps = expert_args.pop('n_timesteps')

    env = make_vec_env(simulation_env, n_envs=n_envs, rng=np.random.default_rng())
    env.seed(seed)
    env.action_space.seed(seed)

    expert_learn_step = n_timesteps if not expert_learn_step else expert_learn_step

    expert = init_agent(
        po_algo=po_algo,
        # policy=policy,
        env=env,
        seed=seed,
        # kwargs
        **expert_args,
    )
    expert.learn(expert_learn_step)  # Note: set to 100000 to train a proficient expert

    rng = np.random.default_rng()

    vec_env = make_vec_env(
        simulation_env,
        n_envs=n_envs,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    )

    rollouts = rollout.rollout(
        expert,
        vec_env,
        # expert.get_env(),
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng,
    )

    learner_args = get_expert_args(po_algo, task_name=simulation_env)

    if "n_envs" in learner_args.keys():
        n_envs = learner_args.pop('n_envs')
    else:
        n_envs = 8

    n_timesteps = learner_args.pop('n_timesteps')
    venv = make_vec_env(simulation_env, n_envs=n_envs, rng=rng)
    # set random seed for the venv for evaluation
    venv.seed(seed)
    venv.action_space.seed(seed)

    learner = init_agent(
        po_algo=po_algo,
        # policy=policy,
        env=venv,
        seed=None,
        # kwargs
        **learner_args
    )

# FIXME: add wrapper here
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )

    gail_trainer = CustomizeTrainer(
        sample_strategy=sample_strategy,
        sample_key="obs",
        demonstrations=rollouts,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=sample_pool_size,
        n_disc_updates_per_round=n_disc_updates_per_round,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        warm_up=warm_up
    )

    learner_rewards_before_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )

    if not learner_learn_step:
        learner_learn_step = n_timesteps

    gail_trainer.train(learner_learn_step)  # Note: set to 300000 for better results
    loss_record = gail_trainer.loss_record
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )
    return learner_rewards_before_training, learner_rewards_after_training, loss_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--simulation_env', type=str,
                        default="seals/CartPole-v0",
                        choices=["seals/CartPole-v0", "seals/MountainCar-v0", "seals/Ant-v0", "seals/HalfCheetah-v0",
                                 "seals/Hopper-v0", "seals/Humanoid-v0", "seals/Walker2d-v0", "seals/Swimmer-v0",
                                 "Reacher-v2", "Adventure-v0", "BipedalWalker-v3"]
                        # Other possible choices: ["Swimmer-v2", "Swimmer-v3", ]
                        # Adventure-v0:
                        # use https://github.com/HumanCompatibleAI/imitation/blob/master/docs/tutorials/5a_train_preference_comparisons_with_cnn.ipynb
                        )
    parser.add_argument('--po_algo', type=str, default="ppo", choices=['ppo', 'trpo'])
    parser.add_argument('--policy', type=str, default='mlp', choices=['mlp', 'cnn'])
    parser.add_argument('--sample_strategy', type=str, default='random',
                        choices=['random', 'kmeans', 'cos_diff', 'cos_sim', 'scheduling'])    # TODO: add scheduling
    parser.add_argument('--demo_batch_size', type=int, default=1024)
    parser.add_argument('--sample_pool_size', type=int, default=2048)
    # training params
    parser.add_argument('--n_disc_updates_per_round', type=int, default=2)
    parser.add_argument('--expert_learn_step', type=int, default=None)
    parser.add_argument('--learner_learn_step', type=int, default=None)
    parser.add_argument('--repeat_times', type=int, default=3)
    parser.add_argument('--warm_up', type=int, default=3)

    args = parser.parse_args()

    final_results = {
        "rewards_before_training": [],
        "rewards_after_training": [],
        "loss": []
    }
    for i in range(args.repeat_times):
        learner_rewards_before_training, learner_rewards_after_training, loss_record = train(
            seed=args.seed + i,
            simulation_env=args.simulation_env,
            policy=args.policy,
            po_algo=args.po_algo,
            expert_learn_step=args.expert_learn_step,
            sample_strategy=args.sample_strategy,
            demo_batch_size=args.demo_batch_size,
            sample_pool_size=args.sample_pool_size,
            n_disc_updates_per_round=args.n_disc_updates_per_round,
            learner_learn_step=args.learner_learn_step,
            warm_up=args.warm_up
        )
        final_results['rewards_before_training'] += learner_rewards_before_training
        final_results['rewards_after_training'] += learner_rewards_after_training
        final_results['loss'].append(loss_record)

    mean_rewards_before_training = np.mean(final_results['rewards_before_training'])
    mean_rewards_after_training = np.mean(final_results['rewards_after_training'])
    std_rewards_before_training = np.std(final_results['rewards_before_training'])
    std_rewards_after_training = np.std(final_results['rewards_after_training'])

    result_metric = {
        "mean_after": float(mean_rewards_after_training),
        "mean_before": float(mean_rewards_before_training),
        "std_after": float(std_rewards_after_training),
        "std_before": float(std_rewards_before_training)
    }
    print(result_metric)

    output_path = f"outputs/{args.simulation_env}/{args.po_algo}-{args.policy}_policy/{args.sample_strategy}-" \
                  f"sample_{args.demo_batch_size}_from_{args.sample_pool_size}-repeat{args.repeat_times}"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    result_file_path = os.path.join(output_path, "result.json")
    fig_file_path = os.path.join(output_path, "hist.png")
    loss_file_path = os.path.join(output_path, "loss.pickle")

    with open(result_file_path, "w", encoding='utf-8') as f:
        json.dump(result_metric, f)

    with open(loss_file_path, "wb") as f:
        pickle.dump(final_results['loss'], f)

    plt.figure(1)
    plt.hist(
        [np.array(final_results['rewards_before_training']), np.array(final_results['rewards_after_training'])],
        label=["untrained", "trained"],
    )
    plt.legend()
    plt.savefig(fig_file_path)