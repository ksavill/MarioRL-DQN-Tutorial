import torch
import torch.nn as nn
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchvision import transforms as T


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

        # Exploration parameters
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 5e5  # Save checkpoint every so many experiences

        # Replay buffer parameters
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(10000, device=torch.device("cpu"))
        )
        self.batch_size = 32

        # Discount factor for TD target
        self.gamma = 0.9

        # Optimizer and loss function for training
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # Learning schedule parameters
        self.burnin = 1e4  # Minimum experiences before training
        self.learn_every = 3  # Learn every n experiences
        self.sync_every = 1e4  # Sync target network every n experiences

        if checkpoint is not None:
            self.load(checkpoint)

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action.
        If state is a tuple (e.g. from a reset in Gym 0.26+), take the first element.
        """
        if isinstance(state, tuple):
            state = state[0].__array__()
        else:
            state = state.__array__()

        # EXPLORE: choose a random action with probability exploration_rate
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # EXPLOIT: choose the best action according to the Q-network
        else:
            state_tensor = torch.tensor(state, device=self.device).unsqueeze(0).float()
            action_values = self.net(state_tensor, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

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

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

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
            )
        )

    def recall(self):
        """
        Retrieve a batch of experiences from memory.
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """
        Compute the TD estimate: Q_online(state, action)
        """
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        Compute the TD target:
          TD_target = reward + (1 - done) * gamma * Q_target(next_state, best_action)
        """
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """
        Update the online Q-network by backpropagating the loss.
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
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

        state, next_state, action, reward, done = self.recall()

        # Compute TD estimate and target
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        # Update Q-network and return the mean Q-value and loss for logging
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss