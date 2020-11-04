import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.algo.behavior_clone import MlpPolicyNet

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU(),
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class ValueNet(nn.Module):
    def __init__(self, inputs_dim=10, code_dim=3, ft_dim=128, activation=F.relu):
        super(ValueNet, self).__init__()
        self.activation = activation
        self.fc_s1 = nn.Linear(inputs_dim, ft_dim)
        self.fc_s2 = nn.Linear(ft_dim, ft_dim)
        self.code_dim = code_dim
        if code_dim is not None:
            self.fc_c1 = nn.Linear(code_dim, ft_dim)
        # self.fc_c1 = nn.Linear(inputc_dim, ft_dim)
        self.fc_sum = nn.Linear(ft_dim, 1)

    def forward(self, state, latent_code):
        output = self.fc_s2(self.activation(self.fc_s1(state), inplace=True))
        if self.code_dim is not None:
            output += self.fc_c1(latent_code)
        final_out = self.fc_sum(self.activation(output, inplace=True))
        return final_out


class CirclePolicy(nn.Module):
    def __init__(self, obs_shape, code_dim, action_space, base=None, base_kwargs=None):
        super(CirclePolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        # if base is None:
        #     if len(obs_shape) == 3:
        #         base = CNNBase
        #     elif len(obs_shape) == 1:
        #         base = MLPBase
        #     else:
        #         raise NotImplementedError

        self.mlp_policy_net = MlpPolicyNet(obs_shape[0], code_dim, **base_kwargs)
        self.mlp_value_net = ValueNet(obs_shape[0], code_dim, **base_kwargs)
        num_outputs = action_space.shape[0]

        # if action_space.__class__.__name__ == "Discrete":
        #     num_outputs = action_space.n
        #     self.dist = Categorical(self.base.output_size, num_outputs)
        # elif action_space.__class__.__name__ == "Box":
        #     num_outputs = action_space.shape[0]
        #     self.dist = DiagGaussian(self.base.output_size, num_outputs)
        # elif action_space.__class__.__name__ == "MultiBinary":
        #     num_outputs = action_space.shape[0]
        #     self.dist = Bernoulli(self.base.output_size, num_outputs)
        # else:
        #     raise NotImplementedError

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        ## now is set to latent code size: 3
        return 3

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        """
        Use rnn_hxs as latent code input
        """
        # value, actor_features = None, None
        # latent_code = rnn_hxs
        value = self.mlp_value_net(inputs, rnn_hxs)
        action = self.mlp_policy_net.select_action(inputs, rnn_hxs, stochastic=True)

        # dist = self.dist(actor_features)

        # if deterministic:
        #     action = dist.mode()
        # else:
        #     action = dist.sample()

        action_log_probs = self.mlp_policy_net.get_log_prob(inputs, rnn_hxs, action)
        # dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value = self.mlp_value_net(inputs, rnn_hxs)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        # pylint: disable=not-callable
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        value = self.mlp_value_net(inputs, rnn_hxs)
        action = self.mlp_policy_net.select_action(inputs, rnn_hxs, stochastic=False)
        action_log_probs = self.mlp_policy_net.get_log_prob(inputs, rnn_hxs, action)
        device = inputs.device
        dist_entropy = torch.tensor([1]).to(
            device
        )  ## for current fixed action std, entropy will be constant.

        return value, action_log_probs, dist_entropy, rnn_hxs


class Posterior(nn.Module):
    def __init__(
        self, inputs_dim=10, inputa_dim=2, ft_dim=128, outputc_dim=3, activation=F.relu
    ):
        super(Posterior, self).__init__()
        self.fc_s1 = nn.Linear(inputs_dim + inputa_dim, ft_dim)
        self.fc_s2 = nn.Linear(ft_dim, ft_dim)
        self.fc_out = nn.Linear(ft_dim, outputc_dim)
        self.softmax = nn.Softmax(dim=1)
        self.activation = activation
        self.criterion_p = nn.CrossEntropyLoss()
        self.iter_num = 0

    def forward(self, state, action):
        input_data = torch.cat((state, action), dim=1)
        output = self.activation(self.fc_s1(input_data), inplace=True)
        output = self.activation(self.fc_s2(output), inplace=True)
        final_out = self.softmax(self.fc_out(output))
        ### p(c|s,a) where c is the latent code
        return final_out

    def update(self, state, action, encodes, optim_posterior, writer=None):
        self.iter_num += 1
        optim_posterior.zero_grad()
        output_c = self.forward(state, action)
        encodes = torch.argmax(encodes, dim=1)
        loss_p = self.criterion_p(output_c, encodes.long())
        loss_p.backward()
        optim_posterior.step()
        if writer is not None:
            ### .item for scalar from tensor to float
            writer.add_scalar("IG/p_loss", loss_p.item(), self.iter_num)
        # print()


class InfoCirclePolicy(nn.Module):
    """Agent for InfoGAIL
    Attributes:
        mlp_policy_net (:obj:`nn.Module`): The policy network in actor-critic
            architecture
        mlp_value_net (:obj:`nn.Module`): The value network in actor-critic
            architecture
        posterior (:obj:`nn.Module`): The posterior network that encodes any
            given (s,a) tuple
    """

    def __init__(self, obs_shape, code_dim, action_space, base=None, base_kwargs=None):
        super(InfoCirclePolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        self.mlp_policy_net = MlpPolicyNet(obs_shape[0], code_dim, **base_kwargs)
        self.mlp_value_net = ValueNet(obs_shape[0], code_dim, **base_kwargs)
        num_outputs = action_space.shape[0]
        self.posterior = Posterior(inputs_dim=10, inputa_dim=2, ft_dim=128)
        self.optim_posterior = torch.optim.Adam(self.posterior.parameters())
        self.posterior_target = Posterior(inputs_dim=10, inputa_dim=2, ft_dim=128)
        # if action_space.__class__.__name__ == "Discrete":
        #     num_outputs = action_space.n
        #     self.dist = Categorical(self.base.output_size, num_outputs)
        # elif action_space.__class__.__name__ == "Box":
        #     num_outputs = action_space.shape[0]
        #     self.dist = DiagGaussian(self.base.output_size, num_outputs)
        # elif action_space.__class__.__name__ == "MultiBinary":
        #     num_outputs = action_space.shape[0]
        #     self.dist = Bernoulli(self.base.output_size, num_outputs)
        # else:
        #     raise NotImplementedError
        self.actor = self.mlp_policy_net
        self.critic = self.mlp_value_net

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        ## now is set to latent code size: 3
        return 3

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, latent_code, masks, deterministic=False):
        """
        Use rnn_hxs as latent code input
        """
        # value, actor_features = None, None
        # latent_code = rnn_hxs
        value = self.mlp_value_net(inputs, latent_code)
        action = self.mlp_policy_net.select_action(inputs, latent_code, stochastic=True)
        output_p = self.posterior_target(inputs, action)
        reward_p = torch.sum(
            torch.log(output_p + 1e-45) * latent_code, dim=1, keepdim=True
        )

        action_log_probs = self.mlp_policy_net.get_log_prob(inputs, latent_code, action)
        # dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, latent_code, reward_p

    def get_value(self, inputs, rnn_hxs, masks):
        value = self.mlp_value_net(inputs, rnn_hxs)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        # pylint: disable=not-callable
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        value = self.mlp_value_net(inputs, rnn_hxs)
        action = self.mlp_policy_net.select_action(inputs, rnn_hxs, stochastic=False)
        action_log_probs = self.mlp_policy_net.get_log_prob(inputs, rnn_hxs, action)
        device = inputs.device
        dist_entropy = torch.tensor([1]).to(
            device
        )  ## for current fixed action std, entropy will be constant.

        return value, action_log_probs, dist_entropy, rnn_hxs
