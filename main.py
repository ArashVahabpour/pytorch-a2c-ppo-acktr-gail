import copy
import glob
import os
import time
from collections import deque

import gym
import gym_sog
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, CirclePolicy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from tqdm.auto import tqdm

from inference import model_inference_env, visualize_gail_trajs
from utilities import to_tensor, get_module_device, onehot
import wandb
from matplotlib import pyplot as plt


def prepare_agent(actor_critic, args, device):
    if args.algo == "a2c":
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
        )
    elif args.algo == "ppo":
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
        )
    elif args.algo == "acktr":
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True
        )
    elif args.algo == "bc":
        agent = algo.BC(epochs=args.bc_epoch, lr=args.lr, eps=args.eps, device=device,)
    return agent


def main():
    args = get_args()
    if args.wandb:
        wandb.init(config=args)
    else:
        wandb.init(mode="disabled")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:2" if args.cuda else "cpu")
    print("------------**********************-----------------------------")

    env = make_vec_envs(
        args.env_name,
        args.seed,
        args.num_processes,
        args.gamma,
        args.log_dir,
        device,
        False,
        radii=[-10, 10, 20],
        no_render=True,
    )

    # actor_critic = Policy(
    #     env.observation_space.shape,
    #     env.action_space,
    #     base_kwargs={'recurrent': args.recurrent_policy})
    # actor_critic.to(device)
    print(
        "------------**********************---initialize model--------------------------"
    )
    code_dim = None
    actor_critic = CirclePolicy(
        env.observation_space.shape, code_dim, env.action_space, base_kwargs={}
    )
    print("------------**********************---load model--------------------------")
    # base_kwargs={'recurrent': args.recurrent_policy})
    # print("----load pretrained model from BC-----")
    # bc_model_path = "./checkpoints/bestbc_model_new_everywhere2.pth"
    # actor_critic.mlp_policy_net.load_state_dict(torch.load(bc_model_path)['state_dict'])
    print("loaded mlp policy net----", actor_critic.mlp_policy_net)
    actor_critic.to(device)
    print(
        "------------**********************----prepare_agent-------------------------"
    )
    agent = prepare_agent(actor_critic, args, device)
    print(args, args.lr)

    if args.gail:
        assert len(env.observation_space.shape) == 1
        discr = gail.Discriminator(
            env.observation_space.shape[0] + env.action_space.shape[0], 100, device
        )
        # file_name = os.path.join(
        #     args.gail_experts_dir, "trajs_{}.pt".format(
        #         args.env_name.split('-')[0].lower()))
        #### change to the new trainig data 9-28
        # file_name = "/home/shared/datasets/gail_experts/trajs_circles_new.pt"
        # file_name = "/home/shared/datasets/gail_experts/trajs_circles_new.pt"
        file_name = "/home/shared/datasets/gail_experts/trajs_circles_mix.pt"

        expert_dataset = gail.ExpertDataset(
            file_name,
            num_trajectories=args.num_traj,
            subsample_frequency=args.subsample_traj,
        )
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last,
        )

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        env.observation_space.shape,
        env.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    obs = env.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    global_step = 0
    for j in tqdm(range(num_updates)):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer,
                j,
                num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr,
            )

        ### generate one set of random latent codes
        fake_z0 = np.random.randint(3, size=args.num_steps)
        latent_code = np.zeros((args.num_steps, 3))
        latent_code[np.arange(args.num_steps), fake_z0] = 1
        latent_code = torch.FloatTensor(latent_code.copy()).to(device)
        ###change to fixed latent code:

        for step in range(args.num_steps):
            global_step += 1
            # Sample actions
            with torch.no_grad():
                # value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                #     rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                #     rollouts.masks[step])
                value, action, action_log_prob, step_latent_code = actor_critic.act(
                    rollouts.obs[step], latent_code[step], rollouts.masks[step]
                )

            # Obser reward and next obs
            obs, reward, done, infos = env.step(action)

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            ## Note this is next state, current latent code and current action
            rollouts.insert(
                obs,
                step_latent_code,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
            )

        print("step action", rollouts.obs[-1], rollouts.actions[-1])
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], step_latent_code, rollouts.masks[-1]
            ).detach()

        if args.gail:
            if j >= 10:
                env.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            # for _ in range(gail_epoch):
            i = 0
            while True:
                # discr_loss = discr.update(
                #     gail_train_loader, rollouts, utils.get_vec_normalize(env)._obfilt
                # )
                discr_loss, expert_loss, policy_loss, discr_fig = discr.update_visualize(
                    gail_train_loader, rollouts, utils.get_vec_normalize(env)._obfilt
                )
                wandb.log({
                    "discr_loss_only": discr_loss,
                    "discr_expert_loss": expert_loss,
                    "discr_policy_loss": policy_loss,
                    "discr_fig": discr_fig
                }, step=global_step + i)
                i += 1
                # if i % 100 == 0:
                #     radii_list = [20.0, 10.0, -10.0]
                #     codes = onehot(np.repeat(np.arange(len(radii_list)), 5), len(radii_list))
                #     num_trajs = len(codes)
                #     flat_state_arr, action_arr = model_inference_env(
                #         actor_critic.mlp_policy_net,
                #         num_trajs,
                #         10,
                #         state_len=5,
                #         radii=radii_list,
                #         codes=codes,
                #     )
                #     flat_state_tensor = to_tensor(
                #         flat_state_arr, get_module_device(discr)
                #     ).flatten(start_dim=0, end_dim=1)
                #     action_tensor = to_tensor(action_arr, get_module_device(discr)).flatten(
                #         start_dim=0, end_dim=1
                #     )
                #     discr_labels = discr(flat_state_tensor, action_tensor).reshape(
                #         flat_state_arr.shape[0], -1, 1
                #     )
                #     posterior_scalar_codes = None
                #     for mode, radius in enumerate(radii_list):
                #         traj_inds = np.where(codes[:, mode])[0]

                #         fig = visualize_gail_trajs(
                #             flat_state_arr=flat_state_arr,
                #             action_arr=action_arr,
                #             # save_path="./imgs/gail_inference/{os.path.basename(checkpoint_path)}_inference_mode_{i}.png",
                #             discr_labels=discr_labels,
                #             inds=traj_inds,
                #             posterior_scalar_codes=posterior_scalar_codes,
                #             fig_title=f"GAIL Trajectories (r={radius}, code={mode})",
                #         )
                #         wandb.log({f"inference/{mode}": fig}, step=global_step+i)

            wandb.log(
                {
                    # "epoch": epoch,
                    "discr_loss": discr_loss,
                },
                step=global_step,
            )

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step],
                    rollouts.actions[step],
                    args.gamma,
                    rollouts.masks[step],
                )

        rollouts.compute_returns(
            next_value,
            args.use_gae,
            args.gamma,
            args.gae_lambda,
            args.use_proper_time_limits,
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        wandb.log(
            {"reward": wandb.Histogram(rollouts.returns.cpu().numpy())},
            step=global_step,
        )

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        ## hard set
        # if (j % args.save_interval == 0
        if (j % 20 == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save(
                [actor_critic, getattr(utils.get_vec_normalize(env), "ob_rms", None)],
                os.path.join(
                    save_path,
                    args.env_name
                    + f"{args.num_traj}_{args.subsample_traj}_bc_mix_mlp_{j}.pt",
                ),
            )

            # inference
            radii_list = [20.0, 10.0, -10.0]
            codes = onehot(np.repeat(np.arange(len(radii_list)), 5), len(radii_list))
            num_trajs = len(codes)
            flat_state_arr, action_arr = model_inference_env(
                actor_critic.mlp_policy_net,
                num_trajs,
                100,
                state_len=5,
                radii=radii_list,
                codes=codes,
            )
            flat_state_tensor = to_tensor(
                flat_state_arr, get_module_device(discr)
            ).flatten(start_dim=0, end_dim=1)
            action_tensor = to_tensor(action_arr, get_module_device(discr)).flatten(
                start_dim=0, end_dim=1
            )
            discr_labels = discr(flat_state_tensor, action_tensor).reshape(
                flat_state_arr.shape[0], -1, 1
            )
            posterior_scalar_codes = None
            for mode, radius in enumerate(radii_list):
                traj_inds = np.where(codes[:, mode])[0]

                fig = visualize_gail_trajs(
                    flat_state_arr=flat_state_arr,
                    action_arr=action_arr,
                    # save_path="./imgs/gail_inference/{os.path.basename(checkpoint_path)}_inference_mode_{i}.png",
                    discr_labels=discr_labels,
                    inds=traj_inds,
                    posterior_scalar_codes=posterior_scalar_codes,
                    fig_title=f"GAIL Trajectories (r={radius}, code={mode})",
                )
                wandb.log({f"inference/{mode}": fig}, step=global_step)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                f"""Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}
Last {len(episode_rewards)} training episodes: mean/median reward {np.mean(episode_rewards):.1f}/{np.median(episode_rewards):.1f}, min/max reward {np.min(episode_rewards):.1f}/{np.max(episode_rewards):.1f}
dist_entropy: {dist_entropy:.3f}, value_loss: {value_loss:.3f}, action_loss: {action_loss:.3f}
"""
            )
        if (
            args.eval_interval is not None
            and len(episode_rewards) > 1
            and j % args.eval_interval == 0
        ):
            ob_rms = utils.get_vec_normalize(env).ob_rms
            evaluate(
                actor_critic,
                ob_rms,
                args.env_name,
                args.seed,
                args.num_processes,
                eval_log_dir,
                device,
            )


if __name__ == "__main__":
    main()
