"""
Modified version for Infogail
"""
import copy
import glob
import os
import time
from collections import deque
from tensorboardX import SummaryWriter

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
from a2c_ppo_acktr.model import Policy, CirclePolicy, InfoCirclePolicy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from utilities import visualize_pts_tb, to_tensor
from inference import model_inference_env

from tqdm.auto import tqdm


def prepare_agent(actor_critic, args, device):
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)
    elif args.algo == 'bc':
        agent = algo.BC(
            epochs=args.bc_epoch,
            lr=args.lr,
            eps=args.eps,
            device=device,
        )
    else:
        raise ValueError(f"Unrecognized args.algo ({args.algo})")
    return agent


def soft_update(local_model, target_model, tau=0.5):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    =======
        local model (PyTorch model): weights will be copied from
        target model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(),
                                        local_model.parameters()):
        #print("target param, local param:", target_param, local_param)
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)

def main(writer):
    """
    Note 1. Policy entropy inherited from GAIL because the constant std is constant so it is ignored here
    """
    args = get_args()
    p_lambda = 1 ### lambda for low bound latent code mutual information
    IL_method = "infogail"



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

    env = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, radii=[-10, 10, 20], no_render=True)

    # actor_critic = Policy(
    #     env.observation_space.shape,
    #     env.action_space,
    #     base_kwargs={'recurrent': args.recurrent_policy})
    # actor_critic.to(device)
    print("------------**********************---initialize model--------------------------")
    code_dim = 3
    actor_critic = InfoCirclePolicy(
        env.observation_space.shape, code_dim,
        env.action_space,
        base_kwargs={})
    
    
    print("------------**********************---load model--------------------------")
        #base_kwargs={'recurrent': args.recurrent_policy})
    print("----load pretrained model from BC-----")
    bc_model_path = "./checkpoints/bestbc_model_fake_mix.pth"
    
    actor_critic.mlp_policy_net.load_state_dict(torch.load(bc_model_path)['state_dict'])
    print("loaded mlp policy net----", actor_critic.mlp_policy_net)
    actor_critic.to(device)
    posterior = actor_critic.posterior
    posterior_target = actor_critic.posterior_target
    print("------------**********************----prepare_agent-------------------------")
    agent = prepare_agent(actor_critic, args, device)
    print(args,args.lr)

    print("========================= create discriminator =========================")
    assert len(env.observation_space.shape) == 1
    discr = gail.Discriminator(
        env.observation_space.shape[0] + env.action_space.shape[0], 100,
        device)
    # file_name = os.path.join(
    #     args.gail_experts_dir, "trajs_{}.pt".format(
    #         args.env_name.split('-')[0].lower()))
    #### change to the new trainig data 9-28

    print("============== create expert dataset for discriminator =================")
    #file_name = "/home/shared/datasets/gail_experts/trajs_circles_new.pt"
    #file_name = "/home/shared/datasets/gail_experts/trajs_circles_new.pt"
    file_name = "/home/shared/datasets/gail_experts/trajs_circles_mix.pt" # 800 traj, 1000 length

    expert_dataset = gail.ExpertDataset(
        file_name, num_trajectories=args.num_traj, subsample_frequency=args.subsample_traj)
    drop_last = len(expert_dataset) > args.gail_batch_size
    gail_train_loader = torch.utils.data.DataLoader(
        dataset=expert_dataset,
        batch_size=args.gail_batch_size,
        shuffle=True,
        drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              env.observation_space.shape, env.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = env.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in tqdm(range(num_updates)):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        ### generate one set of random latent codes
        # fake_z0 = np.random.randint(3, size=args.num_steps)
        # latent_code = np.zeros((args.num_steps, 3))
        # latent_code[np.arange(args.num_steps), fake_z0] = 1
        # latent_code = torch.FloatTensor(latent_code.copy()).to(device)
        fake_z0 = torch.as_tensor(np.random.randint(3, size=(1,)), device=device)
        latent_code = torch.nn.functional.one_hot(fake_z0, num_classes=3)[0].float()
        ###change to fixed latent code:

        # TODO: def roll_one_step(actor_critic, env, rollouts, step)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                # value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                #     rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                #     rollouts.masks[step])
                value, action, action_log_prob, step_latent_code, reward_p  = actor_critic.act(
                    rollouts.obs[step], latent_code,
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = env.step(action)
            
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            ## Note this is next state, current latent code and current action
            # model_inference_env(actor_critic.mlp_policy_net, 1, 1000, 5, [10], render=True)
            rollouts.insert(obs, step_latent_code, action, reward_p,
                            action_log_prob, value, reward, masks, bad_masks)
        
        print("step action", rollouts.obs[-1], rollouts.actions[-1])
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], latent_code, rollouts.masks[-1]
            ).detach()

        print("====================== train the discriminator =========================")
        if j >= 10:
            env.venv.eval()

        gail_epoch = args.gail_epoch
        if j < 10:
            gail_epoch = 100  # Warm up
        for _ in range(gail_epoch):
            discr.update(gail_train_loader, rollouts, utils.get_vec_normalize(env)._obfilt)

        print("======================== train the posterior ===========================")
        #avg_loss_p = 0
        #TODO: make small batch updates for posterior model
        posterior.update(
            rollouts.obs[:-1].reshape(-1, 10), # obs is longer than actions by one extra entry in the end
            rollouts.actions.reshape(-1, 2),
            rollouts.recurrent_hidden_states[1:].reshape(-1, rollouts.recurrent_hidden_states.shape[-1]),
            actor_critic.optim_posterior, writer
        ) 
        
        soft_update(posterior, posterior_target)
        #avg_loss_p+=loss_p.detach().cpu().data.numpy()
        # Note: in this implementation, discr output 1 means expert, 0 means fake. 
        # ##inversed in the paper
        print("========================= calculate the reward =========================")
        for step in range(args.num_steps):
            rollouts.rewards[step] = discr.predict_reward(
                rollouts.obs[step], rollouts.actions[step], args.gamma,
                rollouts.masks[step]) + p_lambda*rollouts.reward_p[step]
            
            if step % 10 == 0:
                # TODO: def write_reward(rollouts, step, iter_number, writer)
                iter_number = j*args.num_steps + step
                writer.add_scalar("IG/d_reward", rollouts.rewards[step, 0, 0], iter_number)
                writer.add_scalar("IG/p_reward", rollouts.reward_p[step, 0, 0], iter_number)
            if step % 100 == 1:
                iter_number = j*args.num_steps + step
                visualize_pts_tb(writer, rollouts.actions[:step, 0,:].detach().cpu().numpy(), 
                        rollouts.recurrent_hidden_states[1:step+1, 0, :].detach().cpu().numpy(), "IG/action", iter=iter_number)
                visualize_pts_tb(writer, rollouts.obs[:step,0, -2:].detach().cpu().numpy(),
                        rollouts.recurrent_hidden_states[1:step+1, 0, :].detach().cpu().numpy(), "IG/state", iter=iter_number)
                visualize_pts_tb(
                    writer, rollouts.obs[:step,0, -2:].detach().cpu().numpy(),
                    actor_critic.posterior_target(
                        rollouts.obs[:step,0,:], rollouts.actions[:step, 0, :]
                    ).detach().cpu().numpy(),
                    "IG/state_posterior", iter=iter_number
                )
                visualize_pts_tb(
                    writer, rollouts.obs[:step,0, -2:].detach().cpu().numpy(),
                    torch.nn.functional.one_hot(
                        discr(
                            rollouts.obs[:step, 0, :], rollouts.actions[:step, 0, :]
                        ),
                        num_classes=2
                    )[0].detach().cpu().numpy(),
                    "IG/state_discr", iter=iter_number
                )

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        writer.add_scalar("IG/actor_vloss",  value_loss,  j*args.num_steps)
        writer.add_scalar("IG/actor_aloss",  action_loss,  j*args.num_steps)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        ## hard set 
        #if (j % args.save_interval == 0
        if (j % 20 == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, IL_method)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save({
                'args': vars(args),
                'actor_critic': actor_critic,
                'discr': discr,
                'ob_rms': getattr(utils.get_vec_normalize(env), 'ob_rms', None)
            }, os.path.join(save_path, args.env_name + f"{args.num_traj}_{args.subsample_traj}_bc_mix_mlp_{j}.pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                f"""Updates {j}, num timesteps {total_num_steps}, FPS {int(total_num_steps / (end - start))}
Last {len(episode_rewards)} training episodes: mean/median reward {np.mean(episode_rewards):.1f}/{np.median(episode_rewards):.1f}, min/max reward {np.min(episode_rewards):.1f}/{np.max(episode_rewards):.1f}
dist_entropy: {dist_entropy:.3f}, value_loss: {value_loss:.3f}, action_loss: {action_loss:.3f}
"""
            )
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(env).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    writer = SummaryWriter()
    main(writer)
    writer.export_scalars_to_json("logs/info_gail.json")
    writer.close()
