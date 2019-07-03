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
    parser.add_argument('--n_agent_steps', type=int, default=10000, help='No. of steps that agent takes in environment')
    parser.add_argument('--agent_gdt_step_period', type=int, default=10)
    parser.add_argument('--lr_agent', type=float, default=1e-2)
    parser.add_argument('--batch_size_agent', type=int, default=32)
    parser.add_argument('--init_var', type=float, default=1, help='Variance of 0-mean Gaussian from which initial values for Q,R,c are drawn')
    return parser.parse_args()


def main():
    args = parse_arguments()
    # logging
    logdir = './logs/'
    writer = SummaryWriter(log_dir=logdir+'{}'.format(args.info))
    # make environment
    env = gym.make('CartPole-v0')
    obs_shape = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    act_shape = 1
    action_space = [act for act in range(env.action_space.n)]
    # make agent
    lqr_reward = LQR(obs_shape, act_shape, args)
    optimizer_agent = optim.Adam(lqr_reward.parameters(), lr=args.lr_agent)
    replay_buffer = ReplayBufferLQR(capacity=10000)
    epsilon = 0.05
    # loss function
    mse_loss = nn.MSELoss()
    # bookkeeping
    ep_return = 0

    # train agent
    obs = env.reset()
    for step in trange(1, args.n_agent_steps+1, desc='Train agent for {} steps'.format(args.n_agent_steps)):
        if random.random() > epsilon:
            act = np.argmax([lqr_reward(obs, act) for act in action_space])
        else:
            act = env.action_space.sample()
        next_obs, r_true, done, _ = env.step(act)
        replay_buffer.push(obs, act, r_true)
        ep_return += r_true
        obs = next_obs

        if done:
            obs = env.reset()
            writer.add_scalar('ep return against step', ep_return, step)
            ep_return = 0

        # lqr gradient step
        if step % args.agent_gdt_step_period == 0:
            batch_size = min(args.batch_size_agent, len(replay_buffer))
            obs_batch, act_batch, rew_batch = replay_buffer.sample(batch_size)
            r_lqr_batch = lqr_reward(obs_batch, act_batch).squeeze()
            rew_batch = torch.tensor(rew_batch).float()
            loss_agent = mse_loss(r_lqr_batch, rew_batch)
            optimizer_agent.zero_grad()
            loss_agent.backward()
            optimizer_agent.step()
            writer.add_scalar('loss agent per step', loss_agent, step)

    env.close()
    writer.close()

if __name__ == '__main__':
    main()