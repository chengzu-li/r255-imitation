from imitation.algorithms.adversarial.gail import GAIL

from customize_buffer import CustomizeReplayBuffer

import dataclasses

from typing import Callable, Iterable, Iterator, Mapping, Optional, Type, overload

import numpy as np
import torch as th
from torch.nn import functional as F

from imitation.data import buffer, rollout, types, wrappers
from imitation.util import logger, networks, util
from imitation.algorithms.adversarial.common import compute_train_stats

import tqdm


class CustomizeTrainer(GAIL):
    def __init__(
            self,
            sample_strategy: str,
            sample_key: str,
            warm_up: int,
            gen_replay_buffer_capacity: Optional[int] = None,
            **kwargs,
    ):
        """Generative Adversarial Imitation Learning.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            demo_batch_size: The number of samples in each batch of expert data. The
                discriminator batch size is twice this number because each discriminator
                batch contains a generator sample for every expert sample.
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. Environment and logger will be set to
                `venv` and `custom_logger`.
            reward_net: a Torch module that takes an observation, action and
                next observation tensor as input, then computes the logits.
                Used as the GAIL discriminator.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.
        """
        super().__init__(**kwargs)
        self.sample_strategy = sample_strategy
        self.sample_key = sample_key

        if gen_replay_buffer_capacity is None:
            gen_replay_buffer_capacity = self.gen_train_timesteps

        self._gen_replay_buffer = CustomizeReplayBuffer(
            sample_strategy=self.sample_strategy,
            sample_key=self.sample_key,
            capacity=gen_replay_buffer_capacity,
            venv=self.venv,
            warm_up=warm_up
        )
        self.loss_record = []

    def _make_disc_train_batches(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
        idx: int
    ) -> Iterator[Mapping[str, th.Tensor]]:
        """Build and return training minibatches for the next discriminator update.

        Args:
            gen_samples: Same as in `train_disc`.
            expert_samples: Same as in `train_disc`.

        Yields:
            The training minibatch: state, action, next state, dones, labels
            and policy log-probabilities.

        Raises:
            RuntimeError: Empty generator replay buffer.
            ValueError: `gen_samples` or `expert_samples` batch size is
                different from `self.demo_batch_size`.
        """
        batch_size = self.demo_batch_size

        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first.",
                )
            # FIXME: add expert samples to sample more diverse generated data
            gen_samples_dataclass = self._gen_replay_buffer.sample(batch_size, expert_samples, idx)
            gen_samples = types.dataclass_quick_asdict(gen_samples_dataclass)

        if not (len(gen_samples["obs"]) == len(expert_samples["obs"]) == batch_size):
            raise ValueError(
                "Need to have exactly `demo_batch_size` number of expert and "
                "generator samples, each. "
                f"(n_gen={len(gen_samples['obs'])} "
                f"n_expert={len(expert_samples['obs'])} "
                f"demo_batch_size={batch_size})",
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        assert batch_size == len(expert_samples["acts"])
        assert batch_size == len(expert_samples["next_obs"])
        assert batch_size == len(gen_samples["acts"])
        assert batch_size == len(gen_samples["next_obs"])

        for start in range(0, batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            # take minibatch slice (this creates views so no memory issues)
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
            gen_batch = {k: v[start:end] for k, v in gen_samples.items()}

            # Concatenate rollouts, and label each row as expert or generator.
            obs = np.concatenate([expert_batch["obs"], gen_batch["obs"]])
            acts = np.concatenate([expert_batch["acts"], gen_batch["acts"]])
            next_obs = np.concatenate([expert_batch["next_obs"], gen_batch["next_obs"]])
            dones = np.concatenate([expert_batch["dones"], gen_batch["dones"]])
            # notice that the labels use the convention that expert samples are
            # labelled with 1 and generator samples with 0.
            labels_expert_is_one = np.concatenate(
                [
                    np.ones(self.demo_minibatch_size, dtype=int),
                    np.zeros(self.demo_minibatch_size, dtype=int),
                ],
            )

            # Calculate generator-policy log probabilities.
            with th.no_grad():
                obs_th = th.as_tensor(obs, device=self.gen_algo.device)
                acts_th = th.as_tensor(acts, device=self.gen_algo.device)
                log_policy_act_prob = self._get_log_policy_act_prob(obs_th, acts_th)
                if log_policy_act_prob is not None:
                    assert len(log_policy_act_prob) == 2 * self.demo_minibatch_size
                    log_policy_act_prob = log_policy_act_prob.reshape(
                        (2 * self.demo_minibatch_size,),
                    )
                del obs_th, acts_th  # unneeded

            obs_th, acts_th, next_obs_th, dones_th = self.reward_train.preprocess(
                obs,
                acts,
                next_obs,
                dones,
            )
            batch_dict = {
                "state": obs_th,
                "action": acts_th,
                "next_state": next_obs_th,
                "done": dones_th,
                "labels_expert_is_one": self._torchify_array(labels_expert_is_one),
                "log_policy_act_prob": log_policy_act_prob,
            }

            yield batch_dict

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Alternates between training the generator and discriminator.

        Every "round" consists of a call to `train_gen(self.gen_train_timesteps)`,
        a call to `train_disc`, and finally a call to `callback(round)`.

        Training ends once an additional "round" would cause the number of transitions
        sampled from the environment to exceed `total_timesteps`.

        Args:
            total_timesteps: An upper bound on the number of transitions to sample
                from the environment during training.
            callback: A function called at the end of every round which takes in a
                single argument, the round number. Round numbers are in
                `range(total_timesteps // self.gen_train_timesteps)`.
        """
        idx = 0

        n_rounds = int(total_timesteps // self.gen_train_timesteps)
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_train_timesteps} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.train_gen(self.gen_train_timesteps)
            for _ in range(self.n_disc_updates_per_round):
                with networks.training(self.reward_train):
                    # switch to training mode (affects dropout, normalization)
                    self.train_disc(idx=idx)
                    idx += 1
            if callback:
                callback(r)
            self.logger.dump(self._global_step)

    def train_disc(
        self,
        *,
        idx: int,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.demo_batch_size` samples. If this argument is not provided, then
                `self.demo_batch_size` expert samples from `self.demo_data_loader` are
                used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.demo_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.

        Returns:
            Statistics for discriminator (e.g. loss, accuracy).
        """
        with self.logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0

            # compute loss
            self._disc_opt.zero_grad()

            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
                idx=idx
            )
            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
                loss = F.binary_cross_entropy_with_logits(
                    disc_logits,
                    batch["labels_expert_is_one"].float(),
                )

                # Renormalise the loss to be averaged over the whole
                # batch size instead of the minibatch size.
                assert len(batch["state"]) == 2 * self.demo_minibatch_size
                loss *= self.demo_minibatch_size / self.demo_batch_size
                self.loss_record.append(float(loss))
                loss.backward()

            # do gradient step
            self._disc_opt.step()
            self._disc_step += 1

            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    batch["labels_expert_is_one"],
                    loss,
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())

        return train_stats

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ) -> None:
        """Trains the generator to maximize the discriminator loss.

        After the end of training populates the generator replay buffer (used in
        discriminator training) with `self.disc_batch_size` transitions.

        Args:
            total_timesteps: The number of transitions to sample from
                `self.venv_train` during training. By default,
                `self.gen_train_timesteps`.
            learn_kwargs: kwargs for the Stable Baselines `RLModel.learn()`
                method.
        """
        if total_timesteps is None:
            total_timesteps = self.gen_train_timesteps
        if learn_kwargs is None:
            learn_kwargs = {}

        with self.logger.accumulate_means("gen"):
            self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                callback=self.gen_callback,
                **learn_kwargs,
            )
            self._global_step += 1

        gen_trajs, ep_lens = self.venv_buffering.pop_trajectories()
        self._check_fixed_horizon(ep_lens)
        gen_samples = rollout.flatten_trajectories_with_rew(gen_trajs)
        self._gen_replay_buffer.store(gen_samples)