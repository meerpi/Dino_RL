"""
TorchRL Training Script for Chrome Dino.

This script manages the PPO training loop using TorchRL primitives.
It handles parallel data collection, buffer management, and the PPO update logic.
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
    Simple CNN architecture for 84x84 grayscale observation.
    Extracts features and splits into Actor and Critic heads.
    """
    def __init__(self, action_dim=3):
        super().__init__()
        
        # 84x84 inputs
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculated feature size: 64 * 7 * 7
        self.feature_size = 64 * 7 * 7
        
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
        
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
    """Creates a TransformedEnv for TorchRL with 84x84 grayscale stacking."""
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
    
    # Actor (Probabilistic)
    actor = ProbabilisticActor(
        module=nn.Sequential(feature_module, TensorDictModule(net.actor, in_keys=["features"], out_keys=["logits"])),
        spec=CompositeSpec(action=dummy_env.action_spec),
        in_keys=["logits"],
        distribution_class=Categorical,
        return_log_prob=True,
    )
    
    # Critic (Value)
    value_op = ValueOperator(
        module=nn.Sequential(feature_module, TensorDictModule(net.critic, in_keys=["features"], out_keys=["state_value"])),
        in_keys=["pixels"],
    )
    
    # Collector & Buffer
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
