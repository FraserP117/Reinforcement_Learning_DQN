import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
Network:
* 3 conv layers
* cv1: 32, 8*8, stride 4
* cv2: 64, 4*4, stride 2
* cv3: 64, 3*3, stride 1
* out: 512
* optim: RMSProp
* loss: MSELoss
'''


class DQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.cv1 = nn.Conv2d(input_dims[0], 32, (8, 8), stride = 4)
        self.cv2 = nn.Conv2d(32, 64, (4, 4), stride = 2)
        self.cv3 = nn.Conv2d(64, 64, (2, 2), stride = 1)

        fc_input_dims = self.calc_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calc_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.cv1(state)
        dims = self.cv2(dims)
        dims = self.cv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.cv1(state))
        conv2 = F.relu(self.cv2(conv1))
        conv3 = F.relu(self.cv3(conv2))
        # conv3 shape is BS * n_filters * H * W
        # must reshape conv3 to BS * num_input_features to pass to fc1
        conv_state = conv3.view(conv3.size()[0], -1) # -conv3.size()[0] = BS, -1 = flatten all other dimensions
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print("Saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))



class DuelingDQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DuelingDQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.cv1 = nn.Conv2d(input_dims[0], 32, (8, 8), stride = 4)
        self.cv2 = nn.Conv2d(32, 64, (4, 4), stride = 2)
        self.cv3 = nn.Conv2d(64, 64, (2, 2), stride = 1)

        fc_input_dims = self.calc_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)

        # Split into the value and advantage streams:
        self.V = nn.Linear(512, 1) # scalar output
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calc_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.cv1(state)
        dims = self.cv2(dims)
        dims = self.cv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.cv1(state))
        conv2 = F.relu(self.cv2(conv1))
        conv3 = F.relu(self.cv3(conv2))
        # conv3 shape is BS * n_filters * H * W
        # must reshape conv3 to BS * num_input_features to pass to fc1
        conv_state = conv3.view(conv3.size()[0], -1) # -conv3.size()[0] = BS, -1 = flatten all other dimensions
        flat1 = F.relu(self.fc1(conv_state))

        # actions = self.fc2(flat1)

        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        print("Saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))


class DoubleDQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DoubleDQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.cv1 = nn.Conv2d(input_dims[0], 32, (8, 8), stride = 4)
        self.cv2 = nn.Conv2d(32, 64, (4, 4), stride = 2)
        self.cv3 = nn.Conv2d(64, 64, (2, 2), stride = 1)

        fc_input_dims = self.calc_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calc_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.cv1(state)
        dims = self.cv2(dims)
        dims = self.cv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.cv1(state))
        conv2 = F.relu(self.cv2(conv1))
        conv3 = F.relu(self.cv3(conv2))
        # conv3 shape is BS * n_filters * H * W
        # must reshape conv3 to BS * num_input_features to pass to fc1
        conv_state = conv3.view(conv3.size()[0], -1) # -conv3.size()[0] = BS, -1 = flatten all other dimensions
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print("Saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))


class DuelingDoubleDQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DuelingDoubleDQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.cv1 = nn.Conv2d(input_dims[0], 32, (8, 8), stride = 4)
        self.cv2 = nn.Conv2d(32, 64, (4, 4), stride = 2)
        self.cv3 = nn.Conv2d(64, 64, (2, 2), stride = 1)

        fc_input_dims = self.calc_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)

        # Split into the value and advantage streams:
        self.V = nn.Linear(512, 1) # scalar output
        self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calc_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.cv1(state)
        dims = self.cv2(dims)
        dims = self.cv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.cv1(state))
        conv2 = F.relu(self.cv2(conv1))
        conv3 = F.relu(self.cv3(conv2))
        # conv3 shape is BS * n_filters * H * W
        # must reshape conv3 to BS * num_input_features to pass to fc1
        conv_state = conv3.view(conv3.size()[0], -1) # -conv3.size()[0] = BS, -1 = flatten all other dimensions
        flat1 = F.relu(self.fc1(conv_state))

        # actions = self.fc2(flat1)

        V = self.V(flat1)
        A = self.A(flat1)

        return V, A

    def save_checkpoint(self):
        print("Saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))
