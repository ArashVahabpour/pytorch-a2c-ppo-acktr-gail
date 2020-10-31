import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os

# from baselines.common.running_mean_std import RunningMeanStd

import argparse
import tempfile
import os.path as osp
import gym
from tqdm.auto import tqdm

from utilities import (
    normal_log_density,
    set_random_seed,
    to_tensor,
    save_checkpoint,
    load_pickle,
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
)
from a2c_ppo_acktr.algo.behavior_clone import (
    PolicyNet,
    create_dataset,
    create_dataloader,
)
from a2c_ppo_acktr.model import CirclePolicy

if __name__ == "__main__":
    # inference_model()
    # trained_model_dir = "/Users/qiujing/Dropbox/Arash_generative/pytorch-a2c-ppo-acktr-gail"
    trained_model_dir = "."
    IL_method = "infogail"
    # checkpoint_path = os.path.join(
    #     trained_model_dir, "trained_models/ppo/Circles-v0500_20_bc_mlp_100.pt")
    # checkpoint_path = os.path.join(
    #      trained_model_dir, "trained_models/ppo/Circles-v05_20_bc_mix_mlp_40.pt")
    # trained_models/infogail/Circles-v0800_20_bc_mix_mlp_75.pt
    checkpoint_path = os.path.join(
        trained_model_dir, "trained_models/infogail/Circles-v0800_20_bc_mix_mlp_75.pt"
    )
    # trained_models/ppo/Circles-v05_20_bc_mix_mlp_40.pt")
    #     trained_model_dir, "trained_models/ppo/Circles-v0500_20_bc_mix_mlp_100.pt")
    # trained_model_dir, "trained_models/ppo/Circles-v050_1_bc_mlp_100.pt")
    # train_data_path = "/home/shared/datasets/gail_experts/trajs_circles.pt"
    # train_data_path = "/home/shared/datasets/gail_experts/trajs_circles_new.pt"
    train_data_path = "/home/shared/datasets/gail_experts/trajs_circles_mix.pt"
    data_dict = torch.load(train_data_path)
    print("loaded training data info:", data_dict["states"].shape)

    train_dataset, val_dataset = create_dataset(
        train_data_path, fake=True, one_hot=True, one_hot_dim=3
    )
    radii = 20
    state_len = 5
    num_trajs = 20  # number of trajectories
    start_state = get_start_state(num_trajs, mode="sample_data", dataset=val_dataset)
    # device="cuda:0"
    print("start state sampled:", start_state)

    # circle_env, _ = generate_circle_env(state_len=state_len, radius=radii, no_render=False)
    # actor_critic = CirclePolicy(circle_env.observation_space.shape, circle_env.action_space, base_kwargs={})
    try:
        actor_critic = torch.load(checkpoint_path, map_location="cpu")[0]
    except:
        actor_critic = torch.load(checkpoint_path, map_location="cpu")["actor_critic"]
    print("print model", checkpoint_path)
    print(actor_critic.mlp_policy_net)
    # *******************-----------------------------*******************
    code_dim = 3
    fake_code = onehot(np.random.randint(code_dim, size=num_trajs), dim=code_dim)
    traj_len = 1000
    model = actor_critic.mlp_policy_net
    model.eval()

    save_fig_path = os.path.join("./imgs/circle/", IL_method, "val_state_gail.png")
    model_infer_vis(
        model, start_state, fake_code, traj_len, save_fig_path=save_fig_path
    )
    ### load BC basic model as comparison
    # bc_model_path = "./checkpoints/bestbc_model_new_everywhere0.pth"
    # actor_critic2 = torch.load(checkpoint_path, map_location='cpu')[0]
    # bc_model_path = "./checkpoints/bestbc_model_new_everywhere.pth"
    # actor_critic.mlp_policy_net.load_state_dict({key: val if key not in  in for key, val in torch.load(bc_model_path)['state_dict'].items()})
    # sub_model_dict = {key: val for key, val in torch.load(bc_model_path)['state_dict'].items()  if key not in ["fc_c1.weight", "fc_c1.bias"]}
    # sub_model_dict = torch.load(bc_model_path)['state_dict']
    # actor_critic.mlp_policy_net.load_state_dict(sub_model_dict)
    # print("action comparison", )
    # *******************--------------Env infer ---------------*******************
    # flat_state_arr, action_arr = model_inference_env(actor_critic.mlp_policy_net, num_trajs, traj_len, state_len=5, radii=[-10, 10, 20])
    # visualize_trajs_new(flat_state_arr, action_arr, "./imgs/circle/gail_env_inference.png")
    flat_state_arr, action_arr = model_inference_env(
        actor_critic.mlp_policy_net,
        num_trajs,
        traj_len,
        state_len=5,
        radii=[-10, 10, 20],
        render=True,
    )
    # visualize_trajs_new(flat_state_arr, action_arr, "./imgs/circle/gail_env_inference_test.png")

    radii_list = [20, 10, -10]
    for radius in radii_list:
        flat_state_arr, action_arr = model_inference_env(
            actor_critic.mlp_policy_net,
            num_trajs,
            traj_len,
            state_len=5,
            radii=[radius],
        )
        # discrimator = model.
        discr_labels = inference_discrimator(discrimator, flat_state_arr)
        visualize_gail_trajs(
            flat_state_arr,
            action_arr,
            f"./imgs/circle/gail_env_inference_{radius}.png",
            radius=radius,
            discr_labels=discr_labels,
        )

    # print(torch.load(checkpoint_path))
    # print(torch.load(checkpoint_path, map_location=torch.device('cpu'))[0])
    # print(torch.load(checkpoint_path, map_location = 'cpu'))
    # print("print model")
    # print(load_model(actor_critic, checkpoint_path))
    # print(torch.load(checkpoint_path)['state_dict'])
    # self.policy.load_state_dict(torch.load(checkpoint_path)['state_dict'])
