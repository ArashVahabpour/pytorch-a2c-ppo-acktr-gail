import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os

# from baselines.common.running_mean_std import RunningMeanStd

import argparse
from tqdm.auto import tqdm

from utilities import (
    get_module_device,
    set_random_seed,
    to_tensor,
    onehot,
    get_logger,
)

from gym_sog.envs.circle_utils import generate_circle_env
from inference import (
    load_model,
    get_start_state,
    model_infer_vis,
    model_inference_env,
    visualize_trajs_new,
    visualize_gail_trajs,
)
from a2c_ppo_acktr.algo.behavior_clone import (
    PolicyNet,
    create_dataset,
    create_dataloader,
)
from a2c_ppo_acktr.model import CirclePolicy


def get_inference_args(*arg):
    parser = argparse.ArgumentParser(
        description="Inference a (info)gail model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", default="gail", choices=["gail", "infogail"], help="model type"
    )
    parser.add_argument(
        "--gail-experts-dir",
        default="./gail_experts",
        help="directory that contains expert demonstrations for gail",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--cuda-deterministic",
        action="store_true",
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="how many training CPU processes to use",
    )
    parser.add_argument(
        "--env-name", default="Circles-v0", help="environment to infer on",
    )
    parser.add_argument(
        "--log-dir", default="/tmp/gym/", help="directory to save agent logs",
    )
    parser.add_argument(
        "--num-traj", type=int, default=20, help="num of traj to inference for"
    )
    # parser.add_argument("--subsample-traj", type=int, default=20, help="num of traj in the training dataset",)

    parser.add_argument(
        "--checkpoint_path",
        help="directory to save agent logs",
        default="./checkpoints/infogail/Circles-v05_20_bc_mix_mlp_75.pt",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    args = parser.parse_args(*arg)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


# if __name__ == "__main__":
if True:
    state_len = 5
    code_dim = 3
    # args = get_inference_args()
    args = get_inference_args("")
    set_random_seed(args.seed, using_cuda=True)

    trained_model_dir = "."
    IL_method = "infogail"
    # checkpoint_path = os.path.join(trained_model_dir, "trained_models/ppo/Circles-v0500_20_bc_mlp_100.pt")
    # checkpoint_path = os.path.join(trained_model_dir, "trained_models/ppo/Circles-v05_20_bc_mix_mlp_40.pt")
    # checkpoint_path = os.path.join(trained_model_dir, "trained_models/infogail/Circles-v0800_20_bc_mix_mlp_75.pt")
    # checkpoint_path = os.path.join(trained_model_dir, "trained_models/infogail/Circles-v0800_20_bc_mix_mlp_75.pt")

    # train_data_path = "/home/shared/datasets/gail_experts/trajs_circles.pt"
    # train_data_path = "/home/shared/datasets/gail_experts/trajs_circles_new.pt"
    train_data_path = "/home/shared/datasets/gail_experts/trajs_circles_mix.pt"

    train_dataset, val_dataset = create_dataset(
        train_data_path, fake=True, one_hot=True, one_hot_dim=3
    )
    num_trajs = 20  # number of trajectories
    start_state = get_start_state(num_trajs, mode="sample_data", dataset=val_dataset)
    # device="cuda:0"
    print("start state sampled:", start_state)

    # circle_env, _ = generate_circle_env(state_len=state_len, radius=radii, no_render=False)
    # actor_critic = CirclePolicy(circle_env.observation_space.shape, circle_env.action_space, base_kwargs={})
    checkpoint_path = args.checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor_critic = checkpoint["actor_critic"]
    discriminator = checkpoint["discr"]

    print("checkpoint path", checkpoint_path)
    # print("policy network", actor_critic.mlp_policy_net)
    # *******************-----------------------------*******************
    fake_code = onehot(np.random.randint(code_dim, size=num_trajs), dim=code_dim)
    traj_len = 100
    model = actor_critic.mlp_policy_net
    model.eval()

    save_fig_path = os.path.join("./imgs/circle/", IL_method, "val_state_gail.png")
    model_infer_vis(
        model, start_state, fake_code, traj_len, save_fig_path=save_fig_path
    )
    # ============================== Env infer ==============================
    # visualize_trajs_new(flat_state_arr, action_arr, "./imgs/circle/gail_env_inference_test.png")

    radii_list = [20.0, 10.0, -10.0]
    # codes = np.random.randint(len(radii_list), size=(num_trajs,))
    codes = onehot(np.repeat(np.arange(len(radii_list)), 5), len(radii_list))
    num_trajs = len(codes)
    traj_list_state, traj_list_action = [], []
    traj_list_discr_labels = []

    flat_state_arr, action_arr = model_inference_env(
        actor_critic.mlp_policy_net,
        num_trajs,
        traj_len,
        state_len=state_len,
        radii=radii_list,
        codes=codes,
    )
    flat_state_tensor = to_tensor(
        flat_state_arr, get_module_device(discriminator)
    ).flatten(start_dim=0, end_dim=1)
    action_tensor = to_tensor(action_arr, get_module_device(discriminator)).flatten(
        start_dim=0, end_dim=1
    )
    discr_labels = discriminator(flat_state_tensor, action_tensor).reshape(
        flat_state_arr.shape[0], -1, 1
    )
    if hasattr(actor_critic, "posterior"):
        posterior_codes = actor_critic.posterior(flat_state_tensor, action_tensor)
        temp, posterior_scalar_codes = posterior_codes.max(dim=1, keepdim=True)
        posterior_codes = (
            (posterior_codes == temp)
            .detach()
            .cpu()
            .numpy()
            .reshape(flat_state_arr.shape[0], -1, posterior_codes.size(1))
        )
        posterior_scalar_codes = (
            posterior_scalar_codes.detach()
            .cpu()
            .numpy()
            .reshape(flat_state_arr.shape[0], -1, 1)
        )
    else:
        posterior_scalar_codes = None

    code_dim = codes.shape[1]
    for mode, radius in enumerate(radii_list):
        traj_inds = np.where(codes[:, mode])[0]

        visualize_gail_trajs(
            flat_state_arr=flat_state_arr,
            action_arr=action_arr,
            save_path="./imgs/gail_inference/{os.path.basename(checkpoint_path)}_inference_mode_{i}.png",
            discr_labels=discr_labels,
            inds=np.arange(5),  # select the trajectories to plot, by default select all
            posterior_scalar_codes=posterior_scalar_codes,
            fig_title=f"GAIL Trajectories (r={radius}, code={mode})",
        )
