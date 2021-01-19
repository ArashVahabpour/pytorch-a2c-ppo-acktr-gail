import numpy as np
import pickle
from utilities import *
from numpy import random
import matplotlib.pyplot as plt
import copy
from tqdm.auto import tqdm
import os

import gym
from gym_sog.envs.circle_utils import generate_one_traj_env, clip_speed, flat_to_nested

from typing import Union, List, Iterable

from torch import nn


def load_model(model, filename):
    if torch.cuda.is_available():

        def map_location(storage, loc):
            return storage.cuda()

    else:
        map_location = "cpu"
    return model.load_state_dict(
        torch.load(filename, map_location=map_location)["state_dict"]
    )


def make_fake_codes(
    lengths: Union[int, List[int], np.ndarray], dim: int = 3, one_hot: bool = True
):
    if type(lengths) == int:
        lengths = torch.ones(lengths)

    distinct_z = np.random.randint(dim, size=len(lengths))
    z = np.repeat(distinct_z, lengths)
    if one_hot:
        return onehot(z, dim)
    else:
        return z


def make_code_generator(
    code_template: Union[torch.Tensor, np.ndarray, None] = None,
    dim: int = None,
    random: bool = True,
):
    if code_template is None:  # repeat None
        if dim is None:

            def generator():
                while True:
                    yield None

        elif random:

            def generator():
                while True:
                    code = np.zeros(dim)
                    code[np.random.randint(dim)] = 1
                    yield code

        else:
            raise ValueError(
                "Cannot give deterministic non-None codes when code_template is None"
            )
    else:
        assert len(code_template.shape) == 1
        if dim is not None:
            assert (
                code_template.shape[0] == dim
            ), f"dim ({dim}) doesn't match code_template.shape {code_template.shape}"
        else:
            dim = code_template.shape[0]
        if random:

            def generator():
                while True:
                    code = np.zeros(dim)
                    code[np.random.randint(dim)] = 1
                    yield code

        else:

            def generator():
                while True:
                    yield code_template

    return generator()


# TODO: use list for state_arr to make this function dimension-agnostic


def model_inference(model, start_state, latent_code, traj_len):
    device = get_module_device(model)
    if latent_code is not None:
        latent_code = to_tensor(latent_code, device)
    state = start_state.clone().detach().to(device)
    num_traj = start_state.shape[0]
    state_arr = (
        torch.from_numpy(np.zeros((num_traj, traj_len, 5, 2))).float().to(device)
    )
    action_arr = torch.from_numpy(np.zeros((num_traj, traj_len, 2))).float().to(device)
    for i in tqdm(range(traj_len)):  # vectorized along the num_traj axis
        state_flat = state.reshape(state.shape[0], -1)
        speed = model(state_flat, latent_code)
        new_state = step(state, speed, mode="nested")
        state_arr[:, i, :, :] = new_state
        action_arr[:, i, :] = speed
        state = new_state
    return state_arr, action_arr


def visualize_trajs(
    state_arr,
    action_arr,
    traj_len,
    sel_indx=[0],
    color="c",
    save_fig_path="info",
    save_fig_title="model",
):
    # one trajectory visualization
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    for j, traj_ind in enumerate(sel_indx):
        plt.plot(
            state_arr[j, :traj_len, -1, 0],
            state_arr[j, :traj_len, -1, 1],
            "*",
            color=color[j],
        )
    # Action visualization
    plt.subplot(1, 2, 2)
    for j, traj_ind in enumerate(sel_indx):
        plt.plot(
            action_arr[j, :traj_len, 0],
            action_arr[j, :traj_len, 1],
            "*",
            color=color[j],
        )

        # for i in range(traj_len):
        #     plt.plot(action_arr[j, i, 0],
        #              action_arr[j, i, 1], "*", color=color[j])
    plt.title(save_fig_title)
    create_dir(os.path.dirname(save_fig_path))
    plt.savefig(save_fig_path)


def visualize_trajs_new(
    flat_state_arr, action_arr, save_fig_path, save_fig_title="model"
):
    from cycler import cycler

    NUM_COLORS = 20
    cm = plt.get_cmap("tab20")
    # colorlist = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    colorlist = list(cm.colors)
    custom_cycler = cycler(color=colorlist)

    # state visualization
    fig, axs = plt.subplots(ncols=2, figsize=(20, 10))
    axs[0].set_aspect("equal", "box")
    axs[1].set_aspect("equal", "box")
    axs[0].set_prop_cycle(custom_cycler)
    axs[1].set_prop_cycle(custom_cycler)
    for i, traj in enumerate(flat_state_arr):
        axs[0].plot(traj[:, -2], traj[:, -1], "-", alpha=0.5, label=str(i))
    # Action visualization
    for i, a_traj in enumerate(action_arr):
        axs[1].plot(a_traj[:, -2], a_traj[:, -1], "-", alpha=0.5, label=str(i))
    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    plt.title(save_fig_title)
    save_dir = os.path.dirname(save_fig_path)
    create_dir(save_dir)
    plt.savefig(save_fig_path)


def model_infer_vis(
    model,
    start_state,
    latent_code,
    traj_len,
    sel_indx=range(20),
    save_fig_path="info",
    save_fig_title="model",
):
    state_arr, action_arr = model_inference(model, start_state, latent_code, traj_len)
    state_arr = state_arr.cpu().detach().numpy()
    action_arr = action_arr.cpu().detach().numpy()
    NUM_COLORS = 20
    cm = plt.get_cmap("gist_rainbow")
    colorlist = [cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)]
    visualize_trajs(
        state_arr,
        action_arr,
        traj_len,
        sel_indx=sel_indx,
        color=colorlist,
        save_fig_path=save_fig_path,
        save_fig_title=save_fig_title,
    )


def get_start_state(
    n: int, state_dim: int = 2, history_len: int = 5, mode: str = "random", dataset=None
):
    if mode == "random":
        start_state = np.random.normal(0, 0.05, (n, history_len, 2))
    elif mode == "sample_data":
        assert dataset is not None, "Dataset must be specified if mode == 'sample_data'"
        sample_inds = np.random.randint(len(dataset), size=n)
        start_state, _, _ = dataset[sample_inds]
        start_state = start_state.reshape(n, history_len, state_dim)
    return start_state


def model_inference_env(
    model: nn.Module,
    num_traj: int,
    traj_len: int,
    state_len: int,
    radii: List[float],
    codes=None,
    noise_level: float = 0.1,
    render: bool = False,
):
    device = get_module_device(model)

    if codes is None:  # fake codes if necessary
        if model.code_dim is None:
            fake_codes = [None] * num_traj
        else:
            fake_codes = onehot(
                np.random.randint(model.code_dim, size=num_traj), dim=model.code_dim
            )
            fake_codes = to_tensor(fake_codes, device)
        codes = fake_codes
    else:  # use the provided codes
        codes = to_tensor(codes, device)
        assert codes.size(0) == num_traj

    states_arr, action_arr = [], []
    for i, code in enumerate(codes):

        def actor(states, radius, max_ac_mag):
            """Closure capturing code"""
            return clip_speed(
                model(to_tensor(states, device), to_tensor(code, device))
                .cpu()
                .detach()
                .numpy(),
                max_ac_mag,
            )

        if code is None:
            radius = np.random.choice(radii)
        else:  # must be one-hot encoded codes
            radius = np.array(radii)[np.flatnonzero(code.cpu().detach().numpy())[0]]
        # print(f"Generating trajectory #{i}: radius = {radius}")

        states, actions, length = generate_one_traj_env(
            traj_len, state_len, radius, actor, noise_level=noise_level, render=render
        )
        states_arr.append(states)
        action_arr.append(actions)

    states_arr = np.stack(states_arr)
    action_arr = np.stack(action_arr)
    # return flat_to_nested(states_arr, state_len=state_len), action_arr
    return states_arr, action_arr


def visualize_gail_trajs(
    flat_state_arr: np.ndarray,
    action_arr: np.ndarray,
    discr_labels: np.ndarray,
    inds: Union[
        np.ndarray, None
    ] = None,  # select the trajectories to plot, by default select all
    save_path: str = None,
    posterior_scalar_codes: np.ndarray = None,  # scalar codes
    fig_title: str = "GAIL Trajectories",
):
    """Visualize trajectories with (Info)GAIL information

    Include the discriminator's (and posterior encoder's for InfoGAIL) results
    in the plot of states and actions for the Circles-v0 environment
    """
    if inds is None:
        inds = np.arange(len(flat_state_arr))
    from cycler import cycler

    cm = plt.get_cmap("tab20")
    # NUM_COLORS = 20
    # colorlist = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    colorlist = list(cm.colors)
    custom_cycler = cycler(color=colorlist)

    if posterior_scalar_codes is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        axs[0].set_title("States colored w/ discriminator labels")
        axs[1].set_title("Actions colored w/ discriminator labels")
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
        axs[0][0].set_title("States colored w/ discriminator labels")
        axs[0][1].set_title("Actions colored w/ discriminator labels")
        axs[1][0].set_title("States colored w/ posterior codes")
        axs[1][1].set_title("Actions colored w/ posterior codes")
    [ax.set_aspect("equal", "box") for ax in axs.flat]
    [ax.set_prop_cycle(custom_cycler) for ax in axs.flat]
    scatter_legend_lists = [[] for i in range(len(axs.flat))]
    scatter_marker_lists = [[] for i in range(len(axs.flat))]
    for i in inds:
        discr_colors = discr_labels[i]
        traj = flat_state_arr[i]
        a_traj = action_arr[i]
        # state visualization
        axs.flat[0].plot(traj[:, -2], traj[:, -1], "-", alpha=0.5, label=str(i))
        scatter = axs.flat[0].scatter(
            traj[:, -2], traj[:, -1], c=discr_colors, alpha=0.5
        )
        marker, legend = scatter.legend_elements()
        scatter_marker_lists[0].extend(marker)
        scatter_legend_lists[0].extend(legend)
        # Action visualization
        axs.flat[1].plot(a_traj[:, -2], a_traj[:, -1], "-", alpha=0.5, label=str(i))
        scatter = axs.flat[1].scatter(
            a_traj[:, -2], a_traj[:, -1], c=discr_colors, alpha=0.5
        )
        marker, legend = scatter.legend_elements()
        scatter_marker_lists[1].extend(marker)
        scatter_legend_lists[1].extend(legend)
        if posterior_scalar_codes is not None:
            posterior_color = posterior_scalar_codes[i]
            axs.flat[2].plot(traj[:, -2], traj[:, -1], "-", alpha=0.5, label=str(i))
            scatter = axs.flat[2].scatter(
                traj[:, -2], traj[:, -1], c=posterior_color, alpha=0.5
            )
            marker, legend = scatter.legend_elements()
            scatter_marker_lists[2].extend(marker)
            scatter_legend_lists[2].extend(legend)

            axs.flat[3].plot(a_traj[:, -2], a_traj[:, -1], "-", alpha=0.5, label=str(i))
            scatter = axs.flat[3].scatter(
                a_traj[:, -2], a_traj[:, -1], c=posterior_color, alpha=0.5
            )
            marker, legend = scatter.legend_elements()
            scatter_marker_lists[3].extend(marker)
            scatter_legend_lists[3].extend(legend)
    [ax.legend() for ax in axs.flat]

    # add scatter legends
    legend_dict_list = [
        dict(zip(legends, markers))
        for legends, markers in zip(scatter_legend_lists, scatter_marker_lists)
    ]
    [ax.legend(d.values(), d.keys()) for ax, d in zip(axs.flat, legend_dict_list)]

    plt.suptitle(fig_title)
    plt.tight_layout()
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        create_dir(save_dir)
        plt.savefig(save_path)
    # plt.show()
    # plt.close()
    return(fig)
