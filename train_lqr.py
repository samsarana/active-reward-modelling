"""Script to train LQR agent in CartPoleContinuous-v0.
   For now, agent has access to true reward.
"""
import math, random, argparse
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from lqr import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info', type=str, default='', help='Tensorboard log is saved in ./logs/*info*')
    parser.add_argument('--n_agent_steps', type=int, default=10000, help='No. of steps that agent takes in environment, in main training loop')
    parser.add_argument('--agent_gdt_step_period', type=int, default=10)   
    return parser.parse_args()


def main():
    args = parse_arguments()
    # logging
    logdir = './logs/'
    # writer1 = SummaryWriter(log_dir=logdir+'{}_pred'.format(args.info))
    # writer2 = SummaryWriter(log_dir=logdir+'{}_true'.format(args.info))
    writer2 = SummaryWriter(log_dir=logdir+'{}'.format(args.info))

    # make environment
    env = gym.make('CartPoleContinuous-v0', ep_end_penalty=-10.0)
    obs_shape = env.observation_space.shape[0] # env.observation_space is Box(4,) and calling .shape returns (4,) [gym can be ugly]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    act_shape = 1
    action_space = [act for act in range(env.action_space.n)]

    # make agent
    lqr_reward = LQR(obs_shape, act_shape)
    optimizer_agent = optim.Adam(lqr_reward.parameters())
    replay_buffer = ReplayBufferLQR(capacity=10000)
    # loss function
    mse_loss = nn.MSELoss()
    epsilon = 0.05

    # bookkeeping
    dummy_ep_length = env.spec.max_episode_steps
    dummy_returns = {'ep': {'true': 0, 'pred': 0, 'true_norm': 0, 'pred_norm': 0},
                    'all': {'true': [], 'pred': [], 'true_norm': [], 'pred_norm': []}}

    # train agent
    obs = env.reset()
    for step in trange(1, args.n_agent_steps+1, desc='Train agent for {} steps'.format(args.n_agent_steps)):
        # TODO need some epsilon exploration?
        if random.random() > epsilon:
            act = np.argmax([lqr_reward(obs, act) for act in action_space]) # type: numpy.int64
        else:
            act = env.action_space.sample()
        next_obs, r_true, _, _ = env.step(act)
        replay_buffer.push(obs, act, r_true)
        dummy_returns['ep']['true'] += r_true
        obs = next_obs

        # log performance after a "dummy" episode has elapsed
        if step % dummy_ep_length == 0 or step == args.n_agent_steps - 1:
            writer2.add_scalar('dummy ep return against step', dummy_returns['ep']['true'], step)
            for key, value in dummy_returns['ep'].items():
                    dummy_returns['all'][key].append(value)
                    dummy_returns['ep'][key] = 0

        # lqr gradient step
        if step % args.agent_gdt_step_period == 0:
            batch_size = min(32, len(replay_buffer))
            obs_batch, act_batch, rew_batch = replay_buffer.sample(batch_size)
            # assert obs_batch.shape == (batch_size, obs_shape)
            # assert act_batch.shape == (batch_size,) # act.shape is 1 so squeezed
            # assert rew_batch.shape == (batch_size,) # again, squeezed
            r_lqr_batch = lqr_reward(obs_batch, act_batch).squeeze()
            rew_batch = torch.tensor(rew_batch).float()
            # assert r_lqr_batch.shape == rew_batch.shape
            loss_agent = mse_loss(r_lqr_batch, rew_batch) # training with GT reward, for now
            optimizer_agent.zero_grad()
            loss_agent.backward()
            optimizer_agent.step()
            writer2.add_scalar('loss agent per step', loss_agent, step)

    env.close()
    writer2.close()

if __name__ == '__main__':
    main()