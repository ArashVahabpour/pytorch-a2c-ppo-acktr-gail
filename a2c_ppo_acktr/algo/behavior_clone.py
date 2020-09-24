'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import optim

# from baselines.common.running_mean_std import RunningMeanStd

import argparse
import tempfile
import os.path as osp
import gym
from tqdm.auto import tqdm

from utilities import normal_log_density, set_random_seed, to_tensor, save_checkpoint, load_pickle, onehot, get_logger
from abc import ABC, abstractmethod
from typing import List
# import tensorflow as tf


# from baselines.gail import mlp_policy
from baselines import bench
# TODO: migrate ffjord's logger to here (and suppress the logging of full source code (or at least suppress the output of that part))
# from baselines import logger
import logging
# from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
# from baselines.common.mpi_adam import MpiAdam
# from baselines.gail.run_mujoco import runner
# from baselines.gail.dataset.mujoco_dset import Mujoco_Dset

from torch.utils.data import Dataset, TensorDataset, DataLoader


class ExpertTrajectory:
    def __init__(self, path):
        # path = "/mnt/SSD3/Qiujing_exp/Imitation_learning/data/three_modes_traj_train_everywhere_static.pkl"
        #exp_data = load_pickle("three_modes_traj.pkl")
        exp_data = load_pickle(path)
        #self.exp_states = np.concatenate([val['state'].reshape(-1,2,5) for val in exp_data.values()])
        self.expert_states = np.concatenate([val['state'].transpose(
            (0, 1, 3, 2)).reshape(-1, 10) for val in exp_data.values()])
        self.expert_actions = np.concatenate(
            [val['action'].reshape(-1, 2) for val in exp_data.values()])
        self.n_transitions = self.expert_actions.shape[0]
        self.mode_names = list(exp_data.keys())
        self.num_step = exp_data[self.mode_names[0]]['state'].shape[1]
        print("mode names", self.mode_names)
        print("number of transitions:", self.n_transitions)
        print("number of steps in each traj:", self.num_step)
        # np.random.seed(2)

    def sample(self, batch_size):
        """
        Option1: generate latent code for each sample action
        Option2: fix latent code for each traj. Sample states together with latent code
        FIXME: why can the latent code be inconsistent across batches?
        """
        indexes = np.sort(np.random.randint(
            0, self.n_transitions, size=batch_size))
        traj_index = indexes//self.num_step
        unique_traj, traj_fre = np.unique(traj_index, return_counts=True)
        num_unique_traj = len(unique_traj)
        fake_z0 = np.random.randint(3, size=num_unique_traj)
        #new_z = np.zeros(batch_size, dtype=int)
        #print("begin t before latent code sampling:", time() -  start_t)

        new_z = np.repeat(fake_z0, traj_fre)
        #print("end t after latent code sampling:", time() -  start_t)
        fake_z = onehot(new_z)

        state = self.expert_states[indexes]
        action = self.expert_actions[indexes]
        #print("sampled data shape", np.array(state).shape, np.array(action).shape)
        #print("end t after sampling:", time() -  start_t)

        return np.array(state), np.array(action), fake_z, indexes


class TrajectoryDataset(TensorDataset):
    def __init__(self, X, y, c, transform=None, device="cpu"):
        # assert X.shape[0] == y.shape[0] == c.shape[0],\
        #     f"Data length is not aligned: X.shape[0] == {X.shape[0]}, "\
        #     f"y.shape[0] == {y.shape[0]}, c.shape[0] == {c.shape[0]}"
        self.transform = transform if transform is not None else lambda x: x
        X = self.transform(X)
        self.X = torch.as_tensor(X, device=device)  # state
        self.y = torch.as_tensor(y, device=device)  # action
        c = torch.as_tensor(c, device=device, dtype=torch.int64).reshape(-1, 1)
        print(X.shape)
        print(y.shape)
        print(c.shape)
        self.c = torch.zeros(X.shape[0], torch.max(
            c), device=device).scatter_(1, c-1, 1)  # one-hot code
        super(TrajectoryDataset, self).__init__(X, y, c)

    # def __getitem__(self, index):
    #     if self.transform is not None:
    #         tmp = self.X[index]
    #         return self.transform(tmp), self.y[index], self.c[index]
    #     return self.X[index], self.y[index], self.c[index]

    # def __len__(self):
    #     return self.X.shape[0]


def train(epoch, net, dataloader, optimizer, criterion, device, writer):
    net.train()
    train_loss = 0
    # dataloader
    num_batch = len(dataloader)
    for batch_idx, (state, action, latent_code) in enumerate(dataloader):

        optimizer.zero_grad()
        state, action, latent_code = to_tensor(state, device),\
            to_tensor(action, device),\
            to_tensor(latent_code, device)

        outputs = net(state, latent_code)
        #print(outputs, state, latent_code)
        loss = criterion(outputs, action)
        loss.backward()
        optimizer.step()
        #print("loss data", loss.data)
        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Loss: %.3f ' % (train_loss/((batch_idx+1)*3)))
            if writer is not None:
                writer.add_scalars(
                    "Loss/BC", {"train_loss": train_loss/((batch_idx+1)*3)}, batch_idx + num_batch * (epoch-1))


class BC():
    def __init__(self, epochs=300, lr=0.0001, eps=1e-5, device="cpu", policy_activation=F.relu,
                 tb_writer=None, validate_freq=1, checkpoint_dir="."):
        self.epochs = epochs
        self.device = device
        self.policy = MlpPolicyNet(
            state_dim=10, code_dim=None, ft_dim=128, activation=policy_activation).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=eps)
        self.criterion = nn.MSELoss()
        self.writer = tb_writer
        self.validate_freq = validate_freq
        self.checkpoint_dir = checkpoint_dir

    def train(self, expert_loader, val_loader):
        best_loss = float("inf")
        for epoch in tqdm(range(self.epochs)):
            print('\nEpoch: %d' % epoch)
            train(epoch, self.policy, expert_loader, self.optimizer,
                  self.criterion, self.device, self.writer)
            if epoch % self.validate_freq == 0:
                best_loss, checkpoint_path = validate(epoch, self.policy, val_loader,
                                                      self.criterion, self.device, best_loss,
                                                      self.writer, self.checkpoint_dir)
        self.load_best_checkpoint(checkpoint_path)

    def load_best_checkpoint(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path)['state_dict'])


def validate(epoch, net, val_loader, criterion, device, best_loss, writer, checkpoint_dir):
    net.eval()
    valid_loss = 0
    number_batches = len(val_loader)
    for batch_idx, (state, action, latent_code) in enumerate(val_loader):
        state, action, latent_code = to_tensor(state, device),\
            to_tensor(action, device),\
            to_tensor(latent_code, device)
        outputs = net(state, latent_code)
        loss = criterion(outputs, action)

        valid_loss += loss.item()

        avg_valid_loss = valid_loss/(batch_idx+1)
        if batch_idx % 100 == 0:
            print('Valid Loss: %.3f ' % (valid_loss/(batch_idx+1)))

            if writer is not None:
                writer.add_scalars("Loss/BC_val", {"val_loss": valid_loss/(
                    (batch_idx+1))}, batch_idx + number_batches * (epoch-1))
    checkpoint_path = osp.join(checkpoint_dir, 'checkpoints/bestbc_model_new_everywhere.pth')
    if avg_valid_loss <= best_loss:
        best_loss = avg_valid_loss
        print('Best epoch: ' + str(epoch))
        save_checkpoint({'epoch': epoch,
                         'avg_loss': avg_valid_loss,
                         'state_dict': net.state_dict(),
                         }, save_path=checkpoint_path)
    return best_loss, checkpoint_path


def load_data(data_file):
    # TODO: remove the radii selection lines
    """For Arash's format
    """
    data_dict = torch.load(data_file)

    X_all = data_dict["states"]
    X_all = X_all[data_dict["radii"] == 10]
    num_traj, traj_len, dim_state = X_all.shape
    X_all = X_all.reshape(-1, dim_state)

    y_all = data_dict["actions"]
    y_all = y_all[data_dict["radii"] == 10]
    num_traj, traj_len, dim_action = y_all.shape
    y_all = y_all.reshape(-1, dim_action)

    c = data_dict["radii"][data_dict["radii"] == 10]
    # change to scalar encoding here in case it's useful
    c_all = torch.zeros(num_traj, traj_len, dtype=torch.int64)
    c_all[c == 10, :] = 1
    c_all[c == 20, :] = 2
    c_all[c == -10, :] = 3
    c_all = c_all.flatten()

    return X_all, y_all, c_all


def create_dataloader(train_data_path, val_data_path=None, batch_size=4):
    #train_data_f = "three_modes_traj.pkl"
    # train_data_path = "three_modes_traj_train_everywhere.pkl"
    # val_data_path = "three_modes_traj_val.pkl"
    from sklearn.model_selection import train_test_split
    X, y, c = load_data(train_data_path)
    if val_data_path is None:
        X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(X, y, c, test_size=0.2)
    else:
        X_train, y_train, c_train = X, y, c
        X_val, y_val, c_val = load_data(val_data_path)

    train_dataset = TrajectoryDataset(X_train, y_train, c_train)
    val_dataset = TrajectoryDataset(X_val, y_val, c_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)
    return train_dataloader, val_dataloader


def get_task_name(args):
    task_name = 'BC'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


class PolicyNet(ABC, nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

    @abstractmethod
    def get_log_prob(self, state, actions):
        pass

    @abstractmethod
    def select_action(self, state, stochastic):
        pass


class MlpPolicyNet(PolicyNet):
    def __init__(self, state_dim=10, code_dim=3, ft_dim=128, activation=F.relu):
        super(MlpPolicyNet, self).__init__()
        self.activation = activation
        self.fc_s1 = nn.Linear(state_dim, ft_dim)
        self.fc_s2 = nn.Linear(ft_dim, ft_dim)
        self.code_dim = code_dim
        if code_dim is not None:
            self.fc_c1 = nn.Linear(code_dim, ft_dim)
        self.fc_sum = nn.Linear(ft_dim, 2)
        self.action_logstds = torch.log(
            torch.from_numpy(np.array([2, 2])).clone().float())
        self.action_std = torch.from_numpy(np.array([2, 2])).clone().float()

        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, state, latent_code):
        output = self.fc_s2(self.activation(self.fc_s1(state), inplace=True))
        if self.code_dim is not None:
            output += self.fc_c1(latent_code)
        final_out = self.fc_sum(self.activation(output, inplace=True))
        return final_out

    def get_log_prob(self, state, latent_code, actions):
        """
        For continus action space. fixed action log std
        """
        action_mu = self.forward(state, latent_code)
        device = state.device
        return normal_log_density(actions, action_mu, self.action_logstds.to(device), self.action_std.to(device))

    def select_action(self, state, latent_code, stochastic=True):
        action_mu = self.forward(state, latent_code)
        #normal_log_density_fixedstd(x, action_mu)
        device = state.device
        if stochastic:
            action = torch.normal(action_mu, self.action_std.to(device))
        else:
            action = action_mu.to(device)
        return action


def argsparser():
    parser = argparse.ArgumentParser(
        "PyTorch Adaption of `baselines` Behavior Cloning")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str,
                        default='data/deterministic.trpo.Hopper.0.00.npz')
    parser.add_argument(
        '--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument(
        '--log_dir', help='the directory to save log file', default='log')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=False,
                 help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False,
                 help='save the trajectories or not')
    parser.add_argument(
        '--BC_max_iter', help='Max iteration for training BC', type=int, default=1e5)
    return parser.parse_args()

# def main(args):
#     U.make_session(num_cpu=1).__enter__()
#     set_global_seeds(args.seed)
#     env = gym.make(args.env_id)

    # def policy_fn(name, ob_space, ac_space, reuse=False):
    #     return MlpPolicyNet()
#     env = bench.Monitor(env, logger.get_dir() and
#                         osp.join(logger.get_dir(), "monitor.json"))
#     env.seed(args.seed)
#     gym.logger.setLevel(logging.WARN)
#     task_name = get_task_name(args)
#     args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
#     args.log_dir = osp.join(args.log_dir, task_name)
#     dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
#     savedir_fname = learn(env,
#                           policy_fn,
#                           dataset,
#                           max_iters=args.BC_max_iter,
#                           ckpt_dir=args.checkpoint_dir,
#                           log_dir=args.log_dir,
#                           task_name=task_name,
#                           verbose=True)
#     avg_len, avg_ret = runner(env,
#                               policy_fn,
#                               savedir_fname,
#                               timesteps_per_batch=1024,
#                               number_trajs=10,
#                               stochastic_policy=args.stochastic_policy,
#                               save=args.save_sample,
#                               reuse=True)


if __name__ == '__main__':
    #     args = argsparser()
    #     main(args)
    # train_data_path = "three_modes_traj_train_everywhere.pkl"
    # val_data_path = "three_modes_traj_val.pkl"
    train_data_path = "/home/shared/datasets/gail_experts/trajs_circles.pt"
    bc = BC(epochs=30, lr=1e-4, eps=1e-5, device="cuda:0")
    train_loader, val_loader = create_dataloader(train_data_path, batch_size=400)
    bc.train(train_loader, val_loader)
