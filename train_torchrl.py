"""
TorchRL Training Script for Chrome Dino.

I'm using TorchRL here because it's built on PyTorch (which I know) but adds 
all the RL primitives I'd otherwise have to write from scratch (like buffer management,
parallel data collection, and PPO loss calculation). 
"""

import torch
import numpy as np
import time
import hydra
from dataclasses import dataclass
from torch import nn
from torch.distributions import Categorical

from torchrl.envs import (
    CatFrames,
    GymWrapper,
    Compose,
    StepCounter,
    TransformedEnv,
    ToTensorImage
)
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyMemmapStorage, CompositeSpec
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule

from env_playwright import DinoPlaywrightEnv

# --- Configuration ---
@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Environment
    num_envs: int = 4
    frames_per_batch: int = 1000
    total_frames: int = 1_000_000
    frame_stack: int = 4
    
    # PPO Hyperparameters
    batch_size: int = 256
    num_epochs: int = 10
    lr: float = 3e-4
    gamma: float = 0.99
    lmbda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_eps: float = 1e-4

cfg = Config()


class DinoCNN(nn.Module):
    """
    My custom Neural Network for playing Dino.
    
    It takes the screen as input and outputs two things:
    1. **Actor**: Which button to press (Jump, Duck, or Nothing)?
    2. **Critic**: How "good" is the current situation? (Used for training stability)
    """
    def __init__(self, action_dim=3):
        super().__init__()
        
        # 3-layer CNN to process the 84x84 images.
        # This architecture is based on the famous DQN Nature paper (Mnih et al., 2015).
        # It's surprisingly good at "seeing" simple games like this.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculated feature size: 64 * 7 * 7 = 3136
        self.feature_size = 64 * 7 * 7
        
        # The Actor Head: Outputs logits for 3 actions
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
        
        # The Critic Head: Outputs a single value estimate (V(s))
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )


def make_env(device="cpu"):
    """Creates a TransformedEnv compatible with TorchRL."""
    env = DinoPlaywrightEnv(headless=True)
    env = GymWrapper(env, device=device)
    
def make_env(device="cpu"):
    """
    Factory function to create the environment.
    
    We wrap the base Playwright environment with a few transforms:
    - **ToTensorImage**: Converts raw pixels to PyTorch tensors.
    - **CatFrames**: Stacks 4 frames together. This gives the agent a sense of "motion"
      (velocity/acceleration) that a single static frame can't provide.
    - **StepCounter**: Hard limit on episode length to prevent infinite loops if the agent gets too good.
    """
    env = GymWrapper(DinoPlaywrightEnv(headless=True), device=device)
    return TransformedEnv(env, Compose(
        ToTensorImage(),
        CatFrames(N=4, dim=-3),
        StepCounter(max_steps=10000),
    ))


def main():
    print(f"Training on {cfg.device} with {cfg.num_envs} parallel environments")
    
def main():
    print(f"Training on {cfg.device} with {cfg.num_envs} envs")
    
    # Setup Env & Model
    create_env_fn = lambda: make_env(device=cfg.device)
    dummy_env = create_env_fn()
    
    net = DinoCNN().to(cfg.device)
    
    # Wrap modules for TorchRL (TensorDict flow)
    # Common feature extractor
    feature_module = TensorDictModule(net.feature_extractor, in_keys=["pixels"], out_keys=["features"])
    
    # Actor and Critic heads as TensorDictModules
    policy_head_module = TensorDictModule(net.actor, in_keys=["features"], out_keys=["logits"])
    value_head_module = TensorDictModule(net.critic, in_keys=["features"], out_keys=["state_value"])

    # Sequential containers
    # We pipeline the feature extractor with the respective heads.
    policy_seq = nn.Sequential(feature_module, policy_head_module)
    value_seq = nn.Sequential(feature_module, value_head_module)
    
    # Probabilistic Actor
    # This module handles the sampling logic (e.g., choosing an action based on probabilities).
    actor = ProbabilisticActor(
        module=policy_seq,
        spec=CompositeSpec(action=dummy_env.action_spec),
        in_keys=["logits"],
        distribution_class=Categorical,
        return_log_prob=True,
    )
    
    # Value Operator (Critic)
    # This just outputs the raw value number.
    value_op = ValueOperator(
        module=value_seq,
        in_keys=["pixels"],
    )
    
    # 3. Data Collection
    # The Collector is the workhorse. It runs the policy in the environment 
    # and gathers batches of data. 'SyncDataCollector' means it waits for all
    # envs to finish a batch before returning.
    collector = SyncDataCollector(
        create_env_fn=[create_env_fn] * cfg.num_envs,
        policy=actor,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=cfg.device,
        storing_device=cfg.device,
    )
    
    replay_buffer = ReplayBuffer(storage=LazyMemmapStorage(cfg.frames_per_batch), batch_size=cfg.batch_size)
    
    # Loss & Optimizer
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=value_op,
        clip_epsilon=cfg.clip_epsilon,
        entropy_bonus=True,
        entropy_coef=cfg.entropy_eps,
    )
    advantage_module = GAE(gamma=cfg.gamma, lmbda=cfg.lmbda, value_network=value_op, average_gae=True)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)
    
    # Training Loop
    print("Starting training...")
    start_time = time.time()
    total_frames = 0
    
    for i, data in enumerate(collector):
        total_frames += data.numel()
        
        # Train step
        with torch.no_grad(): advantage_module(data)
        replay_buffer.extend(data.reshape(-1))
        # PPO Update Loop
        # We iterate multiple times over the collected batch to squeeze out as much 
        # learning as possible (that's the "Proximal" part of PPO - making sure we don't change too much).
        for _ in range(cfg.num_epochs):
            for _ in range(cfg.frames_per_batch // cfg.batch_size):
                loss = loss_module(replay_buffer.sample())
                loss_sum = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
                
                optimizer.zero_grad()
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                optimizer.step()
        
        # Logging
        if i % 10 == 0:
            fps = total_frames / (time.time() - start_time)
            print(f"Batch {i}: Frames={total_frames}, FPS={fps:.2f}, Reward={data['next', 'reward'].mean().item():.4f}")
            
    print("Training Complete!")
    torch.save(actor.state_dict(), "models/dino_torchrl.pth")
            
    print("Training Complete!")
    
    torch.save(actor.state_dict(), "models/dino_torchrl.pth")


if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    main()
