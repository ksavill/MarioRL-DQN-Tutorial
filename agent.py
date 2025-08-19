import torch.nn as nn
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchvision import transforms as T
import torch
from torch import amp


class MarioNet(nn.Module):
    """A simple convolutional network.
    
    Architecture:
      - 3 convolutional layers (with ReLU)
      - Flatten
      - 2 fully connected layers (with ReLU) leading to output.
    """

    def __init__(self, input_dim, output_dim):
        super(MarioNet, self).__init__()
        c, h, w = input_dim

        if h != 84 or w != 84:
            raise ValueError(f"Expecting input height and width of 84, got: {h}, {w}")

        # Online network: predicts Q-values
        self.online = self.__build_cnn(c, output_dim)
        # Target network: used to compute the TD target
        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())
        # Freeze target network parameters
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model="online"):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        else:
            raise ValueError("model must be 'online' or 'target'")

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Device: {self.device}")

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Build the Q-network
        self.net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)

        # Enable cudnn autotuner for convs on GPU
        try:
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # Mixed precision for faster training on GPU
        self.use_amp = (device == "cuda")
        self.scaler = amp.GradScaler('cuda', enabled=self.use_amp)

        # Exploration parameters
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e5  # Save checkpoint every so many experiences

        # Replay buffer parameters - use CPU memmap storage; move to GPU on sample
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(10000)
        )
        self.batch_size = 128

        # Discount factor for TD target
        self.gamma = 0.9

        # Optimizer and loss function for training
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # Learning schedule parameters
        self.burnin = 2000  # Minimum experiences before training
        self.learn_every = 1  # Learn every n experiences
        self.sync_every = 1e4  # Sync target network every n experiences
        self.updates_per_step = 4  # Number of gradient updates per env step
        
        # GPU optimization settings
        self.prefetch_factor = 2 if self.device.type == 'cuda' else 1
        self.persistent_workers = self.device.type == 'cuda'

        if checkpoint is not None:
            self.load(checkpoint)

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action.
        If state is a tuple (e.g. from a reset in Gym 0.26+), take the first element.
        """
        # unwrap tuple from gym>=0.26 reset() and ensure channel-first (C,H,W)
        if isinstance(state, tuple):
            state = state[0]
        state = state.__array__()
        # FrameStack returns (H,W,C) where C=4; convert to (C,H,W)
        if state.ndim == 3 and state.shape[2] == 4:
            state = np.transpose(state, (2, 0, 1)).copy()

        # EXPLORE: choose a random action with probability exploration_rate
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # EXPLOIT: choose the best action according to the Q-network
        else:
            # Ensure contiguous memory to avoid negative/irregular strides
            state = np.ascontiguousarray(state)
            # Use pinned memory for faster CPU-GPU transfer
            state_tensor = torch.from_numpy(state).pin_memory().to(self.device, non_blocking=True).unsqueeze(0).float()
            with torch.no_grad():  # No gradients needed for inference
                action_values = self.net(state_tensor, model="online")
                action_idx = torch.argmax(action_values, dim=1).item()

        # Decay exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience (state, action, reward, next_state, done) in memory.
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state)
        next_state = first_if_tuple(next_state)
        state = state.__array__()
        next_state = next_state.__array__()
        # Ensure channel-first for storage
        if state.ndim == 3 and state.shape[2] == 4:
            state = np.transpose(state, (2, 0, 1)).copy()
        if next_state.ndim == 3 and next_state.shape[2] == 4:
            next_state = np.transpose(next_state, (2, 0, 1)).copy()

        # Store CPU tensors in replay buffer; transfer to GPU at sampling time
        state = torch.from_numpy(np.ascontiguousarray(state)).float()
        next_state = torch.from_numpy(np.ascontiguousarray(next_state)).float()
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        done = torch.tensor([done], dtype=torch.bool)

        self.memory.add(
            TensorDict(
                {
                    "state": state,
                    "next_state": next_state,
                    "action": action,
                    "reward": reward,
                    "done": done,
                },
                batch_size=[],
                device=torch.device('cpu'),
            )
        )

    def recall(self):
        """
        Retrieve a batch of experiences from memory.
        """
        # Sample from CPU storage and move batch to target device
        batch = self.memory.sample(self.batch_size)
        if self.device.type == 'cuda':
            batch = batch.to(self.device, non_blocking=True)
        state, next_state, action, reward, done = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """
        Compute the TD estimate: Q_online(state, action)
        """
        batch_index = torch.arange(0, self.batch_size, device=self.device)
        current_Q = self.net(state, model="online")[batch_index, action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        Compute the TD target:
          TD_target = reward + (1 - done) * gamma * Q_target(next_state, best_action)
        """
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        batch_index = torch.arange(0, self.batch_size, device=self.device)
        next_Q = self.net(next_state, model="target")[batch_index, best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """
        Update the online Q-network by backpropagating the loss.
        """
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            with amp.autocast('cuda'):
                loss = self.loss_fn(td_estimate, td_target)
            self.scaler.scale(loss).backward()
            # Gradient clipping for stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            return loss.item()
        else:
            loss = self.loss_fn(td_estimate, td_target)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            return loss.item()

    def sync_Q_target(self):
        """
        Copy the parameters from the online network to the target network.
        """
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        """
        Save the current network and exploration rate as a checkpoint.
        """
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            {"model": self.net.state_dict(), "exploration_rate": self.exploration_rate},
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, checkpoint):
        """
        Load a checkpoint.
        """
        data = torch.load(checkpoint)
        self.net.load_state_dict(data["model"])
        self.exploration_rate = data["exploration_rate"]

    def learn(self):
        """
        Sample a batch from memory and update the Q-network.
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        mean_q = 0.0
        mean_loss = 0.0
        for _ in range(int(self.updates_per_step)):
            state, next_state, action, reward, done = self.recall()
            # Compute TD estimate and target
            td_est = self.td_estimate(state, action)
            td_tgt = self.td_target(reward, next_state, done)
            # Update Q-network
            loss = self.update_Q_online(td_est, td_tgt)
            mean_q += td_est.mean().item()
            mean_loss += loss
        mean_q /= float(self.updates_per_step)
        mean_loss /= float(self.updates_per_step)
        return mean_q, mean_loss