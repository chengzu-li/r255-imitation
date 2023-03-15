import torch
import imitation

from collections import OrderedDict

from imitation.policies.base import NormalizeFeaturesExtractor

def get_expert_args(po_algo: str, task_name: str):
    task_name = task_name.split("/")[-1]
    if task_name == "Hopper-v0":
        # TODO: 2048
        agent_args = OrderedDict([('batch_size', 512),
             ('clip_range', 0.1),
             ('ent_coef', 0.0010159833764878474),
             ('gae_lambda', 0.98),
             ('gamma', 0.995),
             ('learning_rate', 0.0003904770450788824),
             ('max_grad_norm', 0.9),
             ('n_envs', 1),
             ('n_epochs', 20),
             ('n_steps', 2048),
             ('n_timesteps', 100000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs',
              {'activation_fn': torch.nn.modules.activation.ReLU,
               'features_extractor_class': 'imitation.policies.base.NormalizeFeaturesExtractor',
               'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]}),
             ('vf_coef', 0.20315938606555833),
             ])

    elif task_name == "CartPole-v0":
        # TODO: 2048
        agent_args = OrderedDict([('batch_size', 256),
             ('clip_range', 0.4),
             ('ent_coef', 0.008508727919228772),
             ('gae_lambda', 0.9),
             ('gamma', 0.9999),
             ('learning_rate', 0.0012403278189645594),
             ('max_grad_norm', 0.8),
             ('n_epochs', 10),
             ('n_steps', 512),
             ('n_envs', 4),
             ('n_timesteps', 100000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs',
              {'activation_fn': torch.nn.modules.activation.ReLU,
               'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]}),
             ('vf_coef', 0.489343896591493)])
    elif task_name == "Ant-v0":
        # TODO: 2048
        agent_args = OrderedDict([('batch_size', 16),
             ('clip_range', 0.3),
             ('ent_coef', 3.1441389214159857e-06),
             ('gae_lambda', 0.8),
             ('gamma', 0.995),
             ('learning_rate', 0.00017959211641976886),
             ('max_grad_norm', 0.9),
             ('n_epochs', 10),
             ('n_steps', 2048),
             ('n_envs', 1),
             ('n_timesteps', 1000000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs',
              {'activation_fn': torch.nn.modules.activation.Tanh,
               'features_extractor_class': NormalizeFeaturesExtractor,
               'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]}),
             ('vf_coef', 0.4351450387648799),
             ])
    elif task_name == "HalfCheetah-v0":
        # TODO: 2048
        agent_args = OrderedDict([('batch_size', 64),
             ('clip_range', 0.1),
             ('ent_coef', 3.794797423594763e-06),
             ('gae_lambda', 0.95),
             ('gamma', 0.95),
             ('learning_rate', 0.0003286871805949382),
             ('max_grad_norm', 0.8),
             ('n_envs', 4),
             ('n_epochs', 5),
             ('n_steps', 512),
             ('n_timesteps', 1000000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs',
              {'activation_fn': torch.nn.modules.activation.Tanh,
               'features_extractor_class': NormalizeFeaturesExtractor,
               'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]}),
             ('vf_coef', 0.11483689492120866),
             ])
    elif task_name == "Humanoid-v0":
        # TODO: 2048
        agent_args = OrderedDict([('batch_size', 256),
             ('clip_range', 0.2),
             ('ent_coef', 2.0745206045994986e-05),
             ('gae_lambda', 0.92),
             ('gamma', 0.999),
             ('learning_rate', 2.0309225666232827e-05),
             ('max_grad_norm', 0.5),
             ('n_envs', 1),
             ('n_epochs', 20),
             ('n_steps', 2048),
             ('n_timesteps', 100000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs',
              {'activation_fn': torch.nn.modules.activation.ReLU,
               'features_extractor_class': NormalizeFeaturesExtractor,
               'net_arch': [{'pi': [256, 256], 'vf': [256, 256]}]}),
             ('vf_coef', 0.819262464558427),
             ])
    elif task_name == "Walker2d-v0":
        # TODO: 2048
        agent_args = OrderedDict([('batch_size', 8),
             ('clip_range', 0.4),
             ('ent_coef', 0.00013057334805552262),
             ('gae_lambda', 0.92),
             ('gamma', 0.98),
             ('learning_rate', 3.791707778339674e-05),
             ('max_grad_norm', 0.6),
             ('n_envs', 1),
             ('n_epochs', 5),
             ('n_steps', 2048),
             ('n_timesteps', 100000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs',
              {'activation_fn': torch.nn.modules.activation.ReLU,
               'features_extractor_class': NormalizeFeaturesExtractor,
               'net_arch': [{'pi': [256, 256], 'vf': [256, 256]}]}),
             ('vf_coef', 0.6167177795726859),
             ])
    elif task_name == "MountainCar-v0":
        # TODO: 2048
        agent_args = OrderedDict([('batch_size', 512),
             ('clip_range', 0.2),
             ('ent_coef', 6.4940755116195606e-06),
             ('gae_lambda', 0.98),
             ('gamma', 0.99),
             ('learning_rate', 0.0004476103728105138),
             ('max_grad_norm', 1),
             ('n_envs', 8),
             ('n_epochs', 20),
             ('n_steps', 256),
             ('n_timesteps', 1000000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs',
              {'activation_fn': torch.nn.modules.activation.Tanh,
               'features_extractor_class': NormalizeFeaturesExtractor,
               'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]}),
             ('vf_coef', 0.25988158989488963),
             ])
    elif task_name == "Swimmer-v0":
        # TODO: 2048
        agent_args = OrderedDict([('batch_size', 8),
             ('clip_range', 0.1),
             ('ent_coef', 5.167107294612664e-08),
             ('gae_lambda', 0.95),
             ('gamma', 0.999),
             ('learning_rate', 0.0001214437022727675),
             ('max_grad_norm', 2),
             ('n_epochs', 20),
             ('n_steps', 2048),
             ('n_envs', 1),
             ('n_timesteps', 100000.0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs',
              {'activation_fn': torch.nn.modules.activation.Tanh,
               'features_extractor_class': NormalizeFeaturesExtractor,
               'net_arch': [{'pi': [64, 64], 'vf': [64, 64]}]}),
             ('vf_coef', 0.6162112311062333),
             ])

    elif task_name == "Reacher-v2":
        agent_args = {
            "batch_size": 64,
            "ent_coef": 0.0,
            "learning_rate": 0.0003,
            "n_epochs": 10,
            "n_steps": 64,
            "n_envs": 8,
            "n_timesteps": 1000000,
            "policy": "MlpPolicy"
        }
    elif task_name == "Adventure-v0":
        agent_args = {
            "batch_size": 64,
            "ent_coef": 0.0,
            "learning_rate": 0.0003,
            "n_epochs": 10,
            "n_steps": 64,
            "n_envs": 8,
            "n_timesteps": 1000000,
            "policy": "MlpPolicy"
        }
    else:
        raise NotImplementedError

    return agent_args