"""
Inference script that loads ONLY the policy weights from a Windows-trained model.
Workaround for cross-platform pickle incompatibility between Python 3.11 and 3.13.
"""

import os
import sys
import torch
import zipfile
import io
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import gymnasium as gym

from env_playwright import DinoPlaywrightEnv


class ChannelFirstWrapper(gym.ObservationWrapper):
    """Wrapper to convert (H, W) observations to (1, H, W) format."""
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(1, old_shape[0], old_shape[1]),
            dtype=np.uint8
        )

    def observation(self, obs):
        return np.expand_dims(obs, axis=0)


def make_env(headless: bool = False):
    """Create wrapped environment for inference."""
    def _init():
        env = DinoPlaywrightEnv(headless=headless)
        env = ChannelFirstWrapper(env)
        return env
    return _init


def load_model_weights_only(model_path: str, env, device='cpu'):
    """
    Load only the policy weights from a model zip file.
    This bypasses the pickle compatibility issues.
    """
    # Create a new PPO model with the same architecture
    model = PPO(
        "CnnPolicy",
        env,
        device=device,
        verbose=0
    )
    
    # Extract and load policy weights from the zip
    with zipfile.ZipFile(model_path, 'r') as zf:
        with zf.open('policy.pth') as f:
            buffer = io.BytesIO(f.read())
            policy_state = torch.load(buffer, map_location=device, weights_only=False)
    
    # Load the state dict into the model's policy
    model.policy.load_state_dict(policy_state)
    print(f"✅ Loaded policy weights from {model_path}")
    
    return model


def test_model(model_path: str, num_episodes: int = 5, headless: bool = False):
    """Load and test the trained PPO model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Headless: {headless}")
    print(f"{'='*60}\n")
    
    # Create environment with frame stacking
    print("Creating environment...")
    env = DummyVecEnv([make_env(headless=headless)])
    env = VecFrameStack(env, n_stack=4)
    
    # Load model with weights-only approach
    print(f"Loading model weights from {model_path}...")
    model = load_model_weights_only(model_path, env, device=device)
    
    # Run episodes
    total_scores = []
    total_rewards = []
    
    for ep in range(num_episodes):
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---")
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info = env.step(action)
            
            episode_reward += reward[0]
            steps += 1
            done = done_vec[0]
            
            if steps % 50 == 0:
                score = info[0].get('score', 0)
                print(f"  Step {steps}: Score={score}, Reward={episode_reward:.2f}")
        
        final_score = info[0].get('score', 0)
        total_scores.append(final_score)
        total_rewards.append(episode_reward)
        
        print(f"\n✅ Episode {ep + 1} Complete!")
        print(f"   Final Score: {final_score}")
        print(f"   Total Reward: {episode_reward:.2f}")
        print(f"   Steps: {steps}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {np.mean(total_scores):.1f}")
    print(f"Max Score: {max(total_scores)}")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"{'='*60}\n")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained Dino PPO model")
    parser.add_argument('--model', '-m', type=str, default='models/dino_ppo_final.zip')
    parser.add_argument('--episodes', '-e', type=int, default=3)
    parser.add_argument('--headless', action='store_true')
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model,
        num_episodes=args.episodes,
        headless=args.headless
    )
