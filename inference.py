import numpy as np
import pickle
from utilities import *
from numpy import random
import matplotlib.pylab as plt
import copy
from tqdm.auto import tqdm

import gym
from gym_sog.envs.circle_utils import generate_one_traj_env, clip_speed, flat_to_nested

from typing import Union, List
from numbers import Real


def load_model(model, filename):
    if torch.cuda.is_available():
        def map_location(storage, loc): return storage.cuda()
    else:
        map_location = 'cpu'
    return model.load_state_dict(torch.load(filename, map_location=map_location)['state_dict'])


def fake_codes(lengths: Union[int, List[int], np.ndarray], dim: int = 3, one_hot: bool = True):
    if type(lengths) == int:
        lengths = torch.ones(lengths)

    distinct_z = np.random.randint(dim, size=len(lengths))
    z = np.repeat(distinct_z, lengths)
    if one_hot:
        return onehot(z, dim)
    else:
        return z

# TODO: use list for state_arr to make this function dimension-agnostic


def model_inference(model, start_state, latent_code, traj_len):
    device = get_module_device(model)
    latent_code = to_tensor(latent_code, device)
    state = start_state.clone().detach().to(device)
    num_traj = start_state.shape[0]
    state_arr = torch.from_numpy(
        np.zeros((num_traj, traj_len, 5, 2))).float().to(device)
    action_arr = torch.from_numpy(
        np.zeros((num_traj, traj_len, 2))).float().to(device)
    for i in tqdm(range(traj_len)):  # vectorized along the num_traj axis
        state_flat = state.reshape(state.shape[0], -1)
        speed = model(state_flat, latent_code)
        new_state = step(state, speed, mode="nested")
        state_arr[:, i, :, :] = new_state
        action_arr[:, i, :] = speed
        state = new_state
    return state_arr, action_arr


def visualize_trajs(state_arr, action_arr, traj_len, sel_indx=0,
                    color="c", save_fig_name="info", save_fig_title="model"):
    # one trajectory visualization
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    for j, traj_ind in enumerate(sel_indx):
        for i in range(traj_len):
            plt.plot(state_arr[j, i, -1, 0],
                     state_arr[j, i, -1, 1], "*", color=color[j])
    # Action visualization
    plt.subplot(1, 2, 2)
    for j, traj_ind in enumerate(sel_indx):
        for i in range(traj_len):
            plt.plot(action_arr[j, i, 0],
                     action_arr[j, i, 1], "*", color=color[j])
    plt.title(save_fig_title)
    save_dir = "./imgs/circle/"
    create_dir(save_dir)
    plt.savefig(os.path.join(save_dir, f'{save_fig_name}.png'))


def visualize_trajs_new(flat_state_arr, action_arr, save_fig_path, save_fig_title="model"):
    from cycler import cycler
    NUM_COLORS = 20
    cm = plt.get_cmap('gist_rainbow')
    colorlist = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    custom_cycler = cycler(color=colorlist)

    # state visualization
    fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
    axs[0].set_aspect('equal', 'box'); axs[1].set_aspect('equal', 'box')
    axs[0].set_prop_cycle(custom_cycler); axs[1].set_prop_cycle(custom_cycler)
    for i, traj in enumerate(flat_state_arr):
        axs[0].plot(traj[:,-2], traj[:, -1], "*", label=str(i))
    # Action visualization
    for i, a_traj in enumerate(action_arr):
        axs[1].plot(a_traj[:,-2], a_traj[:, -1], "*", label=str(i))
    plt.legend()
    plt.tight_layout()
    plt.title(save_fig_title)
    save_dir = os.path.dirname(save_fig_path)
    create_dir(save_dir)
    plt.savefig(save_fig_path)


def model_infer_vis(model, start_state, latent_code, traj_len,
                    sel_indx=range(20), save_fig_name="info", save_fig_title="model"):
    state_arr, action_arr = model_inference(
        model, start_state, latent_code, traj_len)
    state_arr = state_arr.cpu().detach().numpy()
    action_arr = action_arr.cpu().detach().numpy()
    NUM_COLORS = 20
    cm = plt.get_cmap('gist_rainbow')
    colorlist = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    visualize_trajs(state_arr, action_arr, traj_len, sel_indx=sel_indx,
                    color=colorlist, save_fig_name=save_fig_name, save_fig_title=save_fig_title)


def get_start_state(n: int, state_dim: int = 2, history_len: int = 5,
                    mode: str = "random", dataset=None):
    if mode == "random":
        start_state = np.random.normal(0, 0.05, (n, history_len, 2))
    elif mode == "sample_data":
        assert dataset is not None, "Dataset must be specified if mode == 'sample_data'"
        sample_inds = np.random.randint(len(dataset), size=n)
        start_state, _, _ = dataset[sample_inds]
        start_state = start_state.reshape(n, history_len, state_dim)
    return start_state


def model_inference_env(model, num_traj: int, traj_len: int, state_len: int, radii: List[Real]):
    device = get_module_device(model)

    if model.code_dim is None:
        fake_code = [None] * num_traj
    else:
        fake_code = onehot(np.random.randint(
            model.code_dim, size=num_traj), dim=model.code_dim)
        fake_code = to_tensor(fake_code, device)

    def actor(states, radius, max_ac_mag):
        return clip_speed(
            model(to_tensor(states, device), code).cpu().detach().numpy(),
            max_ac_mag
        )
    states_arr, action_arr = [], []
    for i, code in enumerate(fake_code):
        radius = np.random.choice(radii)
        print(f"Generating trajectory #{i}: radius = {radius}")


        states, actions, length = generate_one_traj_env(
            traj_len, state_len, radius, actor, noise=True)
        states_arr.append(states)
        action_arr.append(actions)

    states_arr = np.stack(states_arr)
    action_arr = np.stack(action_arr)
    # return flat_to_nested(states_arr, state_len=state_len), action_arr
    return states_arr, action_arr
