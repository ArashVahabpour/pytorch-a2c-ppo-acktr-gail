import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import resolve_latent_code
from itertools import cycle


class PPO():
    def __init__(self, actor_critic, args, use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.bc_coeff = args.bc_coeff

        self.latent_size = args.latent_size

        self.max_grad_norm = args.max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr, eps=args.eps)

        self.mse = nn.MSELoss()
        self.device = args.device

    def update(self, expert_loader, rollouts, actor_critic, obsfilt):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        bc_loss_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for expert_batch, policy_batch in zip(cycle(expert_loader), data_generator):
                obs_batch, latent_code_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = policy_batch

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, latent_code_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                expert_state, expert_action = expert_batch
                expert_state = obsfilt(expert_state.numpy(), update=False)
                expert_state = torch.FloatTensor(expert_state).to(self.device)
                expert_action = expert_action.to(self.device)
                expert_latent_code = resolve_latent_code(actor_critic, expert_state, expert_action, self.latent_size)
                expert_action_pred = self.actor_critic.act(expert_state, expert_latent_code, None, deterministic=True)[1]
                bc_loss = self.mse(expert_action_pred, expert_action)

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef + bc_loss * self.bc_coeff).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                bc_loss_epoch += bc_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        bc_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, bc_loss_epoch
