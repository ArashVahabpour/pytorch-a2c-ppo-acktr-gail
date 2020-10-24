import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
import os
import os.path as osp

#sys.path.insert(0, "../../..")
#print(sys.path)
from .behavior_clone import MlpPolicyNet, create_dataset
from .gail import ExpertDataset
from inference import load_model, get_start_state, model_infer_vis, model_inference_env, visualize_trajs_new
from utilities import to_tensor, save_checkpoint, onehot


USE_CUDA = True

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20, mode="Traj"):
        """
        Mode: 
        1. Trajectory based: each sample will return one traj of (state, action) pairs
        2. State based: each sample will return (state, action) pair
        """
        self.mode = mode
        all_trajectories = torch.load(file_name)
        
        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k == 'radii':
                continue
            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.length = self.trajectories['lengths'].sum().item()

        if mode == "state":
            for i in range(num_trajectories):
                self.get_idx.append((i, i))
        else:
            traj_idx = 0
            i = 0
            self.get_idx = []
            
            for j in range(self.length):
                
                # when `i` grows beyond one of the trajectories, increment the `traj_idx` and accordingly set back `i`
                while self.trajectories['lengths'][traj_idx].item() <= i:
                    i -= self.trajectories['lengths'][traj_idx].item()
                    traj_idx += 1

                self.get_idx.append((traj_idx, i))
                i += 1


    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.mode == "state":
            traj_idx, i = self.get_idx[i]

            return self.trajectories['states'][traj_idx][i], self.trajectories[
                'actions'][traj_idx][i]

        traj_idx, _ = self.get_idx[i]

        return self.trajectories['states'][traj_idx][:], self.trajectories[
            'actions'][traj_idx][:]


def create_train_val_split(dataset, batch_size=16, shuffle=True, validation_split=0.1):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    random_seed = 1
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader

"""
class Encoder(nn.Module):
    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        if USE_CUDA:
            eps = eps.cuda()
        std = torch.exp(logvar / 2.0)
        return mu + eps * std
"""
#class Latentcode_sampler(nn.Module):
## refer: https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if USE_CUDA:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, latent_dim = 1, categorical_dim = 3, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3, n_layers=1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.activation=F.relu
        # For LSTM input: (seq_len, batch, input_size)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional)
        if self.bidirectional:
            self.h2z = nn.Linear(hidden_size*2, output_size)
        else:
            self.h2z = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        outputs, (h_n, c_n) = self.lstm(input)
        # outputs shape: (seq_len, batch, num_directions * hidden_size) 
        #print("output shape from bidirectional LSTM:")
        latent_code = self.activation(self.h2z(outputs[-1,:,:]))

        #latent_code
        #mu, logvar = torch.chunk(ps, 2, dim=1)
        z = gumbel_softmax(latent_code, temperature=0.05, latent_dim = 1, categorical_dim = 3, hard=False)
        #z = self.sample(mu, logvar)
        return latent_code, z

class VAE_BC(nn.Module):
    def __init__(self, epochs=300, lr=0.0001, eps=1e-5, device="cpu", tb_writer=None,\
         validate_freq=1, checkpoint_dir=".", code_dim=3, input_size_sa=12, input_size_state=10, hidden_size=128):
        super(VAE_BC, self).__init__()
        self.epochs = epochs
        self.device = device
        self.writer = tb_writer
        self.validate_freq = validate_freq

        self.encoder = EncoderRNN(input_size_sa, hidden_size=hidden_size).to(device)
        self.decoder = MlpPolicyNet(state_dim=input_size_state, code_dim=code_dim, ft_dim=hidden_size).to(device)
        
        self.checkpoint_dir = checkpoint_dir
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters())+ list(self.decoder.parameters()), lr=lr, eps=eps)

    def forward(self, inputs, temperature=1.0):
        """
        Encode the whole trajectory into one latent code. 
        Decode each state with given latent code into action.
        """
        latent_code, z = self.encoder(inputs)
        #n_steps = inputs.size(0)
        input_fts = inputs.size(-1)
        ## reshape as a batch forward
        inputs = inputs.view(-1, input_fts)
        outputs_a = self.mlp_policy_net(torch.cat([inputs, z]))
        return latent_code, z, outputs_a

    def train(self, expert_loader, val_loader):
        best_loss = float("inf")
        for epoch in tqdm(range(self.epochs)):
            print('\nEpoch: %d' % epoch)
            train(epoch, self, expert_loader, self.optimizer,
                  self.criterion, self.device, self.writer)
            if epoch % self.validate_freq == 0:
                best_loss, checkpoint_path = validate(epoch, self, val_loader,
                                                      self.criterion, self.device, best_loss,
                                                      self.writer, self.checkpoint_dir)
        #self.load_best_checkpoint(checkpoint_path)

    def load_best_checkpoint(self, checkpoint_path):
        print("TO BE IMPLEMENTED")
        self.decoder.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        self.encoder.load_state_dict(torch.load(checkpoint_path)['state_dict'])


kld_weight = 0.05 
#TODO: Add adpat weight mechanisms
def validate(epoch, net, val_loader, criterion, device, best_loss, writer, checkpoint_dir):
    net.encoder.eval()
    net.decoder.eval()
    valid_loss = 0
    number_batches = len(val_loader)
    for batch_idx, (traj_state, traj_action) in enumerate(val_loader):
        traj_state = to_tensor(traj_state, device)
        traj_action = to_tensor(traj_action, device)
        reshaped_input = torch.cat([traj_state, traj_action], axis=2).permute(1,0,2)
        latent_code, z= net.encoder(reshaped_input)
        decoded_actions = net.decoder(traj_state, latent_code) 
        loss = criterion(decoded_actions, traj_action)

        valid_loss += loss.item()
        avg_valid_loss = valid_loss/(batch_idx + 1)
        if batch_idx % 2 == 0:
            print('Valid Loss: %.3f ' % (valid_loss/(batch_idx + 1)))
            if writer is not None:
                writer.add_scalars("Loss/VAE_BC_val", {"val_loss": valid_loss/(
                    (batch_idx+1))}, batch_idx + number_batches * (epoch-1))

    checkpoint_path = osp.join(
        checkpoint_dir, 'checkpoints/bestvae_bc_model_new_1.pth')
    if avg_valid_loss <= best_loss:
        best_loss = avg_valid_loss
        print('Best epoch: ' + str(epoch))
        save_checkpoint({'epoch': epoch,
                         'avg_loss': avg_valid_loss,
                         'state_dict_encoder': net.encoder,
                         'state_dict_decoder': net.decoder,
                         }, save_path=checkpoint_path)
    return best_loss, checkpoint_path

def train(epoch, net, dataloader, optimizer, criterion, device, writer):
    net.encoder.train()
    net.decoder.train()
    train_loss = 0
    # dataloader
    num_batch = len(dataloader)
    mode_dim = 3
    for batch_idx, (traj_state, traj_action) in enumerate(dataloader):
        optimizer.zero_grad()
        ## traj_state: (seq_len, batch, hidden_size)
        #tmp_input = torch.cat([traj_state.reshape(-1, 10), traj_action.reshape(-1, 2)], axis=1).unsqueeze(1)
        traj_state = to_tensor(traj_state, device)
        traj_action = to_tensor(traj_action, device)
        reshaped_input = torch.cat([traj_state, traj_action], axis=2).permute(1,0,2)
        #print("input shape:", reshaped_input.shape)
        #reshaped_input = to_tensor(reshaped_input, device)

        latent_code, z= net.encoder(reshaped_input)

        #print(outputs, state, latent_code)
        ##VAE loss and action difference loss. 
        ##
       
        decoded_actions = net.decoder(traj_state, latent_code)
        loss = criterion(decoded_actions, traj_action)
        #KLD = (-0.5 * torch.sum(logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1, 1)).mean().squeeze()
        
        log_ratio = torch.log(latent_code * mode_dim  + 1e-20)
        KLD = torch.sum(latent_code * log_ratio, dim=-1).mean()
        
        loss += KLD * kld_weight

        loss.backward()
        optimizer.step()
        #print("loss data", loss.data)
        train_loss += loss.item()

        if batch_idx % 2 == 0:
            print('Loss: %.3f ' % (train_loss/((batch_idx+1)*3)))
            if writer is not None:
                writer.add_scalars(
                    "Loss/VAE_BC", {"train_loss": train_loss/((batch_idx+1)*3)}, batch_idx + num_batch * (epoch-1))
            


"""
m, l, z, decoded = vae(input, temperature)
if temperature > temperature_min:
    temperature -= temperature_dec
KLD = (-0.5 * torch.sum(l - torch.pow(m, 2) - torch.exp(l) + 1, 1)).mean().squeeze()
loss += KLD * kld_weight

if epoch > kld_start_inc and kld_weight < kld_max:
    kld_weight += kld_inc

"""
        
def test():
    seq_len, batch, input_size, num_directions = 3, 1, 5, 2
    in_data = torch.randint(10, (seq_len, batch, input_size)).float()
    ## output: (seq_len, batch, num_directions * hidden_size) 
    ## h_n of shape (num_layers * num_directions, batch, hidden_size)
    ## c_n of shape (num_layers * num_directions, batch, hidden_size)
    output, (h_n, c_n) = lstm(in_data) 

"""
print("indata", in_data)
print(output, output.shape) 
print(h_n, c_n)
print(h_n.shape, c_n.shape)
"""

#if __name__ == '__main__':
if False:
    writer = SummaryWriter()
    ############### Train ###############
    train_data_path = "three_modes_traj_train_everywhere.pkl"
    val_data_path = "three_modes_traj_val.pkl"
    # bc = BC(epochs=30, lr=1e-4, eps=1e-5, device="cuda:0", code_dim=3)
    bc = VAE_BC(epochs=30, lr=1e-4, eps=1e-5, device="cuda:0", code_dim=None)
    # train_data_path = "/home/shared/datasets/gail_experts/trajs_circles.pt"
    train_data_path = "/home/shared/datasets/gail_experts/trajs_circles_mix.pt"
    #train_dataset, val_dataset = create_dataset(train_data_path, fake=True, one_hot=True, one_hot_dim=3)
    #train_loader, val_loader = create_dataloader(train_dataset, val_dataset, batch_size=400)
    #bc.train(train_loader, val_loader)
    batch_size = 128
    expert_dataset = ExpertDataset(
            train_data_path, num_trajectories=500, subsample_frequency=20)

   
    ## Option1: Use all training data for train datset
    # gail_train_loader = torch.utils.data.DataLoader(
    #             dataset=expert_dataset,
    #             batch_size=batch_size,
    #             shuffle=True,
    #             drop_last=drop_last)
    
    batch_size = 16
    train_loader, val_loader = create_train_val_split(expert_dataset, batch_size=batch_size, shuffle=True, validation_split=0.1)
    drop_last = len(train_loader) > batch_size
    bc.train(train_loader, val_loader)
    model_log_dir = "/mnt/SSD3/Qiujing_exp/pytorch-a2c-ppo-acktr-gail/logs/BC_VAE"
    writer.export_scalars_to_json(os.path.join(model_log_dir, "BC_VAE.json"))
    writer.close()

def load_test():
    #model_dir = "/mnt/SSD3/Qiujing_exp/pytorch-a2c-ppo-acktr-gail/a2c_ppo_acktr/algo/"
    model_dir = "."
    tmp = torch.load(os.path.join(model_dir, "checkpoints/bestvae_bc_model_new_1.pth"))
    #print(tmp.keys())
    tmp_encoder = tmp["state_dict_encoder"]
    tmp_decoder = tmp["state_dict_decoder"]
    #tmp_encoder = tmp["state_dict_encoder"]
    print(tmp_encoder)
    print(tmp_decoder)


#load_test()

def test_policy_inference():
    trained_model_dir = "."
    IL_method = "VAE_BC"
    checkpoint_path = os.path.join(
         trained_model_dir, "checkpoints/bestvae_bc_model_new_1.pth")
    policy_net = torch.load(checkpoint_path, map_location='cpu')["state_dict_decoder"]


    train_data_path = "/home/shared/datasets/gail_experts/trajs_circles_mix.pt"
    data_dict = torch.load(train_data_path)
    print("loaded training data info:",  data_dict["states"].shape)
    _, val_dataset = create_dataset(train_data_path, fake=True, one_hot=True, one_hot_dim=3)

    num_trajs = 20  # number of trajectories
    start_state = get_start_state(
        num_trajs, mode="sample_data", dataset=val_dataset)
    #device="cuda:0"
    #print("start state sampled:", start_state)
    # *******************-----------------------------*******************
    code_dim = 3
    fake_code = onehot(np.random.randint(
        code_dim, size=num_trajs), dim=code_dim)
    traj_len = 1000

    model = policy_net
    model.eval()
    print(model)

    save_fig_dir = os.path.join("./imgs/circle", IL_method)
    model_infer_vis(
        model, start_state, fake_code, traj_len, save_fig_path=os.path.join(save_fig_dir, "val_state.png")
    )
     # *******************--------------Env infer ---------------*******************
    #flat_state_arr, action_arr = model_inference_env(actor_critic.mlp_policy_net, num_trajs, traj_len, state_len=5, radii=[-10, 10, 20])
    #visualize_trajs_new(flat_state_arr, action_arr, "./imgs/circle/gail_env_inference.png")
    flat_state_arr, action_arr = model_inference_env(model, num_trajs, traj_len, state_len=5, radii=[-10,10, 20], render=False)
    visualize_trajs_new(flat_state_arr, action_arr, os.path.join(save_fig_dir, "env_infer.png"))

test_policy_inference()