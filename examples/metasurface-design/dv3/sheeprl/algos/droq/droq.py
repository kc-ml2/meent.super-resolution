from __future__ import annotations

import copy
import os
import warnings
from typing import Any, Dict

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch.optim import Optimizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler
from torchmetrics import SumMetric

from sheeprl.algos.droq.agent import DROQAgent, build_agent
from sheeprl.algos.sac.loss import entropy_loss, policy_loss
from sheeprl.algos.sac.sac import test
from sheeprl.data.buffers import ReplayBuffer
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import Ratio, save_configs


def train(
    fabric: Fabric,
    agent: DROQAgent,
    actor_optimizer: Optimizer,
    qf_optimizer: Optimizer,
    alpha_optimizer: Optimizer,
    rb: ReplayBuffer,
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    per_rank_gradient_steps: int,
):
    # Sample a minibatch in a distributed way: Line 5 - Algorithm 2
    # We sample one time to reduce the communications between processes
    sample = rb.sample_tensors(
        per_rank_gradient_steps * cfg.algo.per_rank_batch_size,
        sample_next_obs=cfg.buffer.sample_next_obs,
        from_numpy=cfg.buffer.from_numpy,
    )
    critic_data: Dict[str, torch.Tensor] = fabric.all_gather(sample)  # [World, G*B]
    for k, v in critic_data.items():
        critic_data[k] = v.float()  # [G*B*World]
        if fabric.world_size > 1:
            critic_data[k] = critic_data[k].flatten(start_dim=0, end_dim=2)
        else:
            critic_data[k] = critic_data[k].flatten(start_dim=0, end_dim=1)
    critic_idxes = range(len(critic_data[next(iter(critic_data.keys()))]))
    if fabric.world_size > 1:
        dist_sampler: DistributedSampler = DistributedSampler(
            critic_idxes,
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=False,
        )
        critic_sampler: BatchSampler = BatchSampler(
            sampler=dist_sampler, batch_size=cfg.algo.per_rank_batch_size, drop_last=False
        )
    else:
        critic_sampler = BatchSampler(sampler=critic_idxes, batch_size=cfg.algo.per_rank_batch_size, drop_last=False)

    # Sample a different minibatch in a distributed way to update actor and alpha parameter
    sample = rb.sample_tensors(cfg.algo.per_rank_batch_size, from_numpy=cfg.buffer.from_numpy)
    actor_data = fabric.all_gather(sample)
    for k, v in actor_data.items():
        actor_data[k] = v.float()  # [G*B*World]
        if fabric.world_size > 1:
            actor_data[k] = actor_data[k].flatten(start_dim=0, end_dim=2)
        else:
            actor_data[k] = actor_data[k].flatten(start_dim=0, end_dim=1)
    if fabric.world_size > 1:
        actor_sampler: DistributedSampler = DistributedSampler(
            range(len(actor_data[next(iter(actor_data.keys()))])),
            num_replicas=fabric.world_size,
            rank=fabric.global_rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=False,
        )
        actor_data = {k: actor_data[k][next(iter(actor_sampler))] for k in actor_data.keys()}

    with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
        # Update the soft-critic
        for batch_idxes in critic_sampler:
            critic_batch_data = {k: critic_data[k][batch_idxes] for k in critic_data.keys()}
            next_target_qf_value = agent.get_next_target_q_values(
                critic_batch_data["next_observations"],
                critic_batch_data["rewards"],
                critic_batch_data["terminated"],
                cfg.algo.gamma,
            )
            for qf_value_idx in range(agent.num_critics):
                # Line 8 - Algorithm 2
                qf_loss = F.mse_loss(
                    agent.get_ith_q_value(
                        critic_batch_data["observations"], critic_batch_data["actions"], qf_value_idx
                    ),
                    next_target_qf_value,
                )
                qf_optimizer.zero_grad(set_to_none=True)
                fabric.backward(qf_loss)
                qf_optimizer.step()
                if aggregator and not aggregator.disabled:
                    aggregator.update("Loss/value_loss", qf_loss)

                # Update the target networks with EMA
                agent.qfs_target_ema(critic_idx=qf_value_idx)

        # Update the actor
        actions, logprobs = agent.get_actions_and_log_probs(actor_data["observations"])
        qf_values = agent.get_q_values(actor_data["observations"], actions)
        min_qf_values = torch.mean(qf_values, dim=-1, keepdim=True)
        actor_loss = policy_loss(agent.alpha, logprobs, min_qf_values)
        actor_optimizer.zero_grad(set_to_none=True)
        fabric.backward(actor_loss)
        actor_optimizer.step()

        # Update the entropy value
        alpha_loss = entropy_loss(agent.log_alpha, logprobs.detach(), agent.target_entropy)
        alpha_optimizer.zero_grad(set_to_none=True)
        fabric.backward(alpha_loss)
        agent.log_alpha.grad = fabric.all_reduce(agent.log_alpha.grad)
        alpha_optimizer.step()

        if aggregator and not aggregator.disabled:
            aggregator.update("Loss/policy_loss", actor_loss)
            aggregator.update("Loss/alpha_loss", alpha_loss)


@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    if "minedojo" in cfg.env.wrapper._target_.lower():
        raise ValueError(
            "MineDojo is not currently supported by DroQ agent, since it does not take "
            "into consideration the action masks provided by the environment, but needed "
            "in order to play correctly the game. "
            "As an alternative you can use one of the Dreamers' agents."
        )

    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size

    # Resume from checkpoint
    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    if len(cfg.algo.cnn_keys.encoder) > 0:
        warnings.warn("DroQ algorithm cannot allow to use images as observations, the CNN keys will be ignored")
        cfg.algo.cnn_keys.encoder = []

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            make_env(
                cfg,
                cfg.seed + rank * cfg.env.num_envs + i,
                rank * cfg.env.num_envs,
                log_dir if rank == 0 else None,
                "train",
                vector_env_idx=i,
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError("Only continuous action space is supported for the DroQ agent")
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")
    if len(cfg.algo.mlp_keys.encoder) == 0:
        raise RuntimeError("You should specify at least one MLP key for the encoder: `mlp_keys.encoder=[state]`")
    for k in cfg.algo.mlp_keys.encoder:
        if len(observation_space[k].shape) > 1:
            raise ValueError(
                "Only environments with vector-only observations are supported by the DroQ agent. "
                f"Provided environment: {cfg.env.id}"
            )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)

    # Define the agent and the optimizer and setup them with Fabric
    agent, player = build_agent(
        fabric, cfg, observation_space, action_space, state["agent"] if cfg.checkpoint.resume_from else None
    )

    # Optimizers
    qf_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=agent.qfs.parameters(), _convert_="all")
    actor_optimizer = hydra.utils.instantiate(
        cfg.algo.actor.optimizer, params=agent.actor.parameters(), _convert_="all"
    )
    alpha_optimizer = hydra.utils.instantiate(cfg.algo.alpha.optimizer, params=[agent.log_alpha], _convert_="all")
    if cfg.checkpoint.resume_from:
        qf_optimizer.load_state_dict(state["qf_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        alpha_optimizer.load_state_dict(state["alpha_optimizer"])
    qf_optimizer, actor_optimizer, alpha_optimizer = fabric.setup_optimizers(
        qf_optimizer, actor_optimizer, alpha_optimizer
    )

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 1
    rb = ReplayBuffer(
        buffer_size,
        cfg.env.num_envs,
        device=device,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], ReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")

    # Global variables
    last_train = 0
    train_step = 0
    start_step = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["update"] // fabric.world_size) + 1
        if cfg.checkpoint.resume_from
        else 1
    )
    policy_step = state["update"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    policy_steps_per_update = int(cfg.env.num_envs * fabric.world_size)
    num_updates = int(cfg.algo.total_steps // policy_steps_per_update) if not cfg.dry_run else 1
    learning_starts = cfg.algo.learning_starts // policy_steps_per_update if not cfg.dry_run else 0
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size
        if not cfg.buffer.checkpoint:
            learning_starts += start_step

    # Create Ratio class
    ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
    if cfg.checkpoint.resume_from:
        ratio.load_state_dict(state["ratio"])

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_update != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )
    if cfg.checkpoint.every % policy_steps_per_update != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_update value ({policy_steps_per_update}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_update value."
        )

    step_data = {}
    # Get the first environment observation and start the optimization
    o = envs.reset(seed=cfg.seed)[0]
    obs = np.concatenate([o[k] for k in cfg.algo.mlp_keys.encoder], axis=-1).astype(np.float32)

    per_rank_gradient_steps = 0
    cumulative_per_rank_gradient_steps = 0
    for update in range(start_step, num_updates + 1):
        policy_step += cfg.env.num_envs * fabric.world_size

        # Measure environment interaction time: this considers both the model forward
        # to get the action given the observation and the time taken into the environment
        with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
            if update <= learning_starts:
                actions = envs.action_space.sample()
            else:
                with torch.inference_mode():
                    # Sample an action given the observation received by the environment
                    actions = player(torch.from_numpy(obs).to(device))
                    actions = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(actions.reshape(envs.action_space.shape))

        if cfg.metric.log_level > 0 and "final_info" in infos:
            for i, agent_ep_info in enumerate(infos["final_info"]):
                if agent_ep_info is not None:
                    ep_rew = agent_ep_info["episode"]["r"]
                    ep_len = agent_ep_info["episode"]["l"]
                    if aggregator and not aggregator.disabled:
                        aggregator.update("Rewards/rew_avg", ep_rew)
                        aggregator.update("Game/ep_len_avg", ep_len)
                    fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

        # Save the real next observation
        real_next_obs = copy.deepcopy(next_obs)
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if final_obs is not None:
                    for k, v in final_obs.items():
                        real_next_obs[k][idx] = v

        next_obs = np.concatenate([next_obs[k] for k in cfg.algo.mlp_keys.encoder], axis=-1).astype(np.float32)
        real_next_obs = np.concatenate([real_next_obs[k] for k in cfg.algo.mlp_keys.encoder], axis=-1).astype(
            np.float32
        )

        step_data["observations"] = obs[np.newaxis]
        if not cfg.buffer.sample_next_obs:
            step_data["next_observations"] = real_next_obs[np.newaxis]
        step_data["actions"] = actions.reshape(1, cfg.env.num_envs, -1)
        step_data["terminated"] = terminated.reshape(1, cfg.env.num_envs, -1).astype(np.float32)
        step_data["truncated"] = truncated.reshape(1, cfg.env.num_envs, -1).astype(np.float32)
        step_data["rewards"] = rewards.reshape(1, cfg.env.num_envs, -1).astype(np.float32)
        rb.add(step_data, validate_args=cfg.buffer.validate_args)

        # next_obs becomes the new obs
        obs = next_obs

        # Train the agent
        if update >= learning_starts:
            per_rank_gradient_steps = ratio(policy_step / world_size)
            if per_rank_gradient_steps > 0:
                train(
                    fabric,
                    agent,
                    actor_optimizer,
                    qf_optimizer,
                    alpha_optimizer,
                    rb,
                    aggregator,
                    cfg,
                    per_rank_gradient_steps,
                )
                train_step += world_size
                cumulative_per_rank_gradient_steps += per_rank_gradient_steps

        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or update == num_updates):
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Log replay ratio
            fabric.log(
                "Params/replay_ratio", cumulative_per_rank_gradient_steps * world_size / policy_step, policy_step
            )

            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics:
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_log) / world_size * cfg.env.action_repeat)
                        / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()

            # Reset counters
            last_log = policy_step
            last_train = train_step

        # Checkpoint model
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            update == num_updates and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "agent": agent.state_dict(),
                "qf_optimizer": qf_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "alpha_optimizer": alpha_optimizer.state_dict(),
                "ratio": ratio.state_dict(),
                "update": update * fabric.world_size,
                "batch_size": cfg.algo.per_rank_batch_size * fabric.world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = os.path.join(log_dir, f"checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt")
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    envs.close()
    if fabric.is_global_zero and cfg.algo.run_test:
        test(player, fabric, cfg, log_dir)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.sac.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {"agent": agent}
        register_model(fabric, log_models, cfg, models_to_log)
