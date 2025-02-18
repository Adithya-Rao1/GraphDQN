import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from replay_buffer import ReplayBuffer
import dqn_hyperparams as dqn_hyperparams
import random
from utils import create_graph, obs_to_loader


class DKDQNNetwork(nn.Module):
    def __init__(self, output_dim=15):
        super(DKDQNNetwork, self).__init__()
        self.gcn1 = None 
        self.gcn2 = GCNConv(128, 512)  
        self.gcn3 = GCNConv(512, 1024)

        self.fc1 = nn.Linear(1024, 512)  
        self.fc2 = nn.Linear(512, 128)  
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, output_dim)

        self.lrelu = nn.LeakyReLU()

    def forward(self, data_batch):
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch  

        if self.gcn1 == None:
            node_feature_dim = x.shape[-1]
            self.gcn1 = GCNConv(node_feature_dim, 128).to(x.device)

        x = self.lrelu(self.gcn1(x, edge_index))
        x = self.lrelu(self.gcn2(x, edge_index))
        x = self.lrelu(self.gcn3(x, edge_index))

        x = global_mean_pool(x, batch)

        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = self.lrelu(self.fc3(x))
        x = self.lrelu(self.fc4(x))
        
        return x

class BootstrapDKDQNNetwork(nn.Module):
    def __init__(self, input_len, output_len=15, num_heads=8):
        super(BootstrapDKDQNNetwork, self).__init__()
        self.num_heads = num_heads
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_len, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
        )

        self.q_heads = nn.ModuleList([
            nn.Linear(32, output_len) for _ in range(num_heads)
        ])

    def forward(self, x, head=None):
        features = self.shared_layers(x)

        if head is not None:
            return self.q_heads[head](features)
        else:
            return torch.stack([head(features) for head in self.q_heads], dim=1)

class DKDQNAgent(object):
    def __init__(self, output_dim, device):
        self.device = device
        self.qn, self.target_qn = (
            DKDQNNetwork(output_dim).to(device),
            DKDQNNetwork(output_dim).to(device),
        )
        for param in self.target_qn.parameters():
            param.requires_grad = False
        self.replay_buffer = ReplayBuffer(dqn_hyperparams.replay_buffer_size)
        self.optimizer = getattr(optim, dqn_hyperparams.optimizer)(self.qn.parameters(), lr=dqn_hyperparams.learning_rate)

    def get_action(self, observations, epsilon_threshold):
        loader = obs_to_loader(observations, batch_size=1)
        
        if np.random.uniform() < epsilon_threshold:
            #print('len of dataset ', len(loader.dataset))
            action = np.random.randint(0, len(loader.dataset))
            #print('chosen action from random exploration, ', action)
        else:
            q_values = []
            for batch in loader:
                q_value = self.target_qn.forward(batch.to(self.device))
                q_values.append(q_value)
            
            q_values = torch.cat(q_values, dim=0)
            q_value = torch.max(q_values, dim=-1)[0]

            action = torch.argmax(q_value).item()
            #print(f'Chosen action from q network: {action}')
        
        return action
    
    def update_params(self, batch_size, gamma, polyak, update_target=False):
        # Sample a batch of transitions
        states, _, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Initialize tensors for Q-values and target Q-values
        q_t = torch.zeros(batch_size, 1, requires_grad=False).to(self.device)
        v_tp1 = torch.zeros(batch_size, 1, requires_grad=False).to(self.device)

        # Convert SMILES to graph data objects and create DataLoader for current and next states
        s_loader = obs_to_loader(create_graph(states), batch_size)
        ns_loader = obs_to_loader(create_graph(next_states), batch_size)

        # Process each batch in the DataLoader for both states and next states
        for batch in s_loader:
            q_t_batch = self.target_qn.forward(batch.to(self.device))
            q_t = torch.max(q_t_batch, dim=1, keepdim=True)[0]  # Assuming q_t is per-state
            
        for batch in ns_loader:
            v_tp1_batch = self.target_qn.forward(batch.to(self.device))
            v_tp1 = torch.max(v_tp1_batch, dim=1, keepdim=True)[0]  # Assuming v_tp1 is per-state
        
        # Convert rewards, done flags to tensors
        rewards = torch.FloatTensor(rewards).reshape(q_t.shape).to(self.device)
        dones = torch.FloatTensor(dones).reshape(q_t.shape).to(self.device)
        
        # Compute the target Q-values using the Bellman equation
        q_tp1_masked = (1 - dones) * v_tp1  # Mask next-state value for terminal states
        q_t_target = rewards + gamma * q_tp1_masked  # Bellman update

        # Compute temporal difference (TD) error
        td_error = q_t - q_t_target

        # Compute the Huber loss (Q-loss)
        q_loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error ** 2,
            1.0 * (torch.abs(td_error) - 0.5),
        )
        q_loss = q_loss.mean()

        # Backpropagation
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        if update_target:
            with torch.no_grad():
                for param, target_param in zip(self.qn.parameters(), self.target_qn.parameters()):
                    target_param.data.mul_(polyak)
                    target_param.data.add_((1 - polyak) * param.data)

        return q_loss

class BootstrapDKDQNAgent(DKDQNAgent):
    def __init__(self, input_len, output_len, device, num_heads=8):
        super(BootstrapDKDQNAgent, self).__init__()
        self.device = device
        self.num_heads = num_heads

        # Initialize networks
        self.qn = BootstrapDKDQNNetwork(input_len, output_len, num_heads).to(device)
        self.target_qn = BootstrapDKDQNNetwork(input_len, output_len, num_heads).to(device)

        # Disable gradient updates for target network
        for param in self.target_qn.parameters():
            param.requires_grad = False

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(dqn_hyperparams.replay_buffer_size)

        # Optimizer
        self.optimizer = getattr(optim, dqn_hyperparams.optimizer)(
            self.qn.parameters(), lr=dqn_hyperparams.learning_rate
        )

    def get_action(self, observations, epsilon_threshold):
        """Select an action using an ε-greedy strategy with a randomly selected Q-head."""
        head_idx = random.randint(0, self.num_heads - 1)  # Choose a random Q-head

        if np.random.uniform() < epsilon_threshold:
            action = np.random.randint(0, dqn_hyperparams.num_actions)
        else:
            with torch.no_grad():
                q_values = self.target_qn.forward(observations.to(self.device), head=head_idx).cpu()
                action = torch.argmax(q_values).numpy()

        return action, head_idx  # Return action and selected head

    def update_params(self, batch_size, gamma, polyak):
        """Update the Q-network using bootstrapped Q-learning."""

        # Sample from replay buffer
        states, _, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        rewards = torch.FloatTensor(rewards).reshape(batch_size, 1).to(self.device)
        dones = torch.FloatTensor(dones).reshape(batch_size, 1).to(self.device)

        q_losses = []
        
        for head_idx in range(self.num_heads):
            q_t = torch.zeros(batch_size, 1, requires_grad=False).to(self.device)
            v_tp1 = torch.zeros(batch_size, 1, requires_grad=False).to(self.device)

            for i in range(batch_size):
                state = torch.FloatTensor(states[i]).to(self.device)
                next_state = torch.FloatTensor(next_states[i]).to(self.device)

                # Q-values from current Q-network head
                q_t[i] = self.qn.forward(state, head=head_idx)

                # Maximum Q-value from target Q-network head
                v_tp1[i] = torch.max(self.target_qn.forward(next_state, head=head_idx))

            # Compute target values
            q_tp1_masked = (1 - dones) * v_tp1
            q_t_target = rewards + gamma * q_tp1_masked

            # Huber loss
            td_error = q_t - q_t_target
            q_loss = torch.where(
                torch.abs(td_error) < 1.0,
                0.5 * td_error**2,
                torch.abs(td_error) - 0.5,
            ).mean()

            q_losses.append(q_loss)

        # Compute overall loss
        q_loss = torch.stack(q_losses).mean()

        # Optimize Q-network
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # Polyak averaging for target network updates
        with torch.no_grad():
            for param, target_param in zip(self.qn.parameters(), self.target_qn.parameters()):
                target_param.data.mul_(polyak)
                target_param.data.add_((1 - polyak) * param.data)

        return q_loss