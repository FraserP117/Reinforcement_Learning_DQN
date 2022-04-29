import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from replay_memory import ReplayBuffer
from dqn import DQNetwork
import pdb

LEARNING_RATE = 0.0001
GAMMA = 0.99
EPS_START = 1.0
EPS_DEC = 5e-7
EPS_MIN = 0.01

class DQNAgent(object):
    def __init__(self, gamma, lr, input_dims, mem_size,
                 batch_size, n_actions, name = None, chkpt_dir = 'tmp/dqn',
                 epsilon = EPS_START, eps_dec = EPS_DEC, eps_min = EPS_MIN,
                 replace_interval = 1000, algo = None, env_name = None):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_interval = replace_interval
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [a for a in range(self.n_actions)]
        self.learn_step_conter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, self.n_actions)

        self.q_eval_net = DQNetwork(
            self.lr, self.n_actions, input_dims = self.input_dims,
            name = self.env_name + '_' + self.algo + '_q_eval',
            chkpt_dir = self.chkpt_dir
        )

        self.q_eval_target_net = DQNetwork(
            self.lr, self.n_actions, input_dims = self.input_dims,
            name = self.env_name + '_' + self.algo + '_q_target',
            chkpt_dir = self.chkpt_dir
        )

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor(np.array([observation]), dtype = T.float).to(self.q_eval_net.device) # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor
            actions = self.q_eval_net.forward(state)
            action = T.argmax(actions).item()

        return action

    def replace_target_network(self):
        if self.learn_step_conter % self.replace_target_interval:
            self.q_eval_target_net.load_state_dict(self.q_eval_net.state_dict())

    def decrement_eps(self):
		# linear decrement
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, next_state, done = \
            self.memory.sample_memory(self.batch_size)

        states = T.tensor(state).to(self.q_eval_net.device)
        actions = T.tensor(action).to(self.q_eval_net.device)
        rewards = T.tensor(reward).to(self.q_eval_net.device)
        next_states = T.tensor(next_state).to(self.q_eval_net.device)
        done_flags = T.tensor(done).to(self.q_eval_net.device)

        return states, actions, rewards, next_states, done_flags

    def save_models(self):
        self.q_eval_net.save_checkpoint()
        self.q_eval_target_net.save_checkpoint()

    def load_models(self):
        self.q_eval_net.load_checkpoint()
        self.q_eval_target_net.load_checkpoint()

    def learn(self, state, action, reward, next_state):
        # filling up the memory with random memories prior to learning anything
        # takes a very long time. We will simply return if the agent has not
        # filled up batch_size memories
        if self.memory.mem_cntr < self.batch_size:
            return

        # zero gradients on the main net and replace the target network:
        self.q_eval_net.optimizer.zero_grad()
        self.replace_target_network()

        # sample the agent's memory:
        states, actions, rewards, next_states, done_flags = self.sample_memory()

        # calculate the q-prediction and q-target values for the actions actually taken:
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval_net.forward(states)[indices, actions] # the action values for the batch of states
        # pdb.set_trace() # have a look at the done mask and q_pred dimensionality
        q_pred_next_states = self.q_eval_target_net.forward(next_states).max(dim = 1)[0] # take the max along the action dimension and take the first element in the tuple (value, index)
        q_pred_next_states[done_flags] = 0.0

        # calculate the td target
        q_target = reward + self.gamma * q_pred_next_states

        # calculate the loss
        loss = self.q_eval_net.loss(q_target, q_pred_next_states).to(self.q_eval_net.device)
        loss.backward()
        self.q_eval_net.optimizer.step()
        self.learn_step_conter += 1

        self.decrement_eps()
