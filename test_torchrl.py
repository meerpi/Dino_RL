"""
Inference Script for TorchRL Dino Agent.

This script loads the trained model (dino_torchrl.pth) and lets it play the game.
It uses the same neural network architecture as training to ensure compatibility.
"""

import torch
import numpy as np
import time
import argparse
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
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.data import CompositeSpec

from env_playwright import DinoPlaywrightEnv


# Maximum number of steps per episode
MAX_STEPS = 10000

class DinoCNN(nn.Module):
    """
    Same architecture as training.
    """
    def __init__(self, action_dim=3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
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


def make_env(headless=False, device="cpu"):
    """
    Creates the environment with the same transforms as training.
    """
    env = GymWrapper(DinoPlaywrightEnv(headless=headless), device=device)
    return TransformedEnv(env, Compose(
        ToTensorImage(),
        CatFrames(N=4, dim=-3),
        StepCounter(max_steps=MAX_STEPS),
    ))


def load_policy(model_path, device, action_spec):
    """
    Reconstructs the agent and loads weights.
    """
    net = DinoCNN().to(device)
    
    # We only need the Actor for inference, but loading the whole state_dict 
    # usually requires the exact same structure if we saved the whole 'actor' module.
    # In training, we saved: torch.save(actor.state_dict(), ...)
    # The 'actor' there was a ProbabilisticActor wrapping a Sequential(feature, policy).
    
    # Reconstruct the exact structure:
    feature_module = TensorDictModule(
        net.feature_extractor, 
        in_keys=["pixels"], 
        out_keys=["features"]
    )
    
    policy_head_module = TensorDictModule(
        net.actor, 
        in_keys=["features"], 
        out_keys=["logits"]
    )
    
    policy_seq = nn.Sequential(feature_module, policy_head_module)
    
    actor = ProbabilisticActor(
        module=policy_seq,
        spec=CompositeSpec(action=action_spec),
        in_keys=["logits"],
        distribution_class=Categorical,
        return_log_prob=True,
    )
    
    # Load weights
    print(f"Loading weights from {model_path} to {device}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(state_dict)
    actor.eval()
    
    return actor


def main():
    parser = argparse.ArgumentParser(description="Watch the Dino Agent Play (TorchRL)")
    parser.add_argument('--model', type=str, default='models/dino_torchrl.pth', help='Path to .pth model file')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to run')
    parser.add_argument('--headless', action='store_true', help='Run without window (faster)')
    parser.add_argument('--device', type=str, default=None, help='Force device (cpu/cuda)')
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create Env
    print(f"Launching environment (headless={args.headless})...")
    env = make_env(headless=args.headless, device=device)
    
    # Load Agent
    try:
        agent = load_policy(args.model, device, env.action_spec)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found.")
        print("Did you run 'python train_torchrl.py' first?")
        return

    # Run Loop
    for i in range(args.episodes):
        print(f"\n--- Episode {i+1}/{args.episodes} ---")
        td = env.reset()
        
        # 'td' is a TensorDict containing the initial observation (on 'device')
        step_count = 0
        total_reward = 0
        
        while not td["done"]:
            # Select action
            with torch.no_grad():
                # The agent writes 'action' into the TensorDict
                agent(td)
                
            # Step the env
            # env.step(td) updates td in-place with next state, reward, done
            td = env.step(td)
            
            # Accumulate reward
            reward = td["next", "reward"].item()
            total_reward += reward
            step_count += 1
            
            # Advance to next state
            td = td["next"]
            
            # Print simple progress
            if step_count % 50 == 0:
                print(f"Step {step_count}: Reward={total_reward:.2f}")

        # Episode finished
        final_score = 0
        # Try to extract score from info if available (TorchRL puts info in a specific place)
        # Our env puts 'score' in info. GymWrapper typically puts info keys in the output TensorDict 
        # as top-level keys if specified, otherwise accessed differently.
        # Since we use simple GymWrapper, 'score' might not be automatically exposed as a tensor unless we registered it.
        # But our custom env returns info dict. GymWrapper wraps this.
        # Check if 'score' is in td.
        if "score" in td.keys():
             final_score = td["score"].item()
        
        print(f"✅ Episode {i+1} Finished!")
        print(f"   Steps: {step_count}")
        print(f"   Total Reward: {total_reward:.2f}")
        # print(f"   Final Score: {final_score}") 

    print("\nDone!")
    env.close()


if __name__ == "__main__":
    main()
