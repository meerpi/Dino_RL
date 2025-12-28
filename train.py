"""
Main training and testing script for a PPO agent to play the Chrome Dino game.

This script handles:
1.  Environment setup: Wraps the custom 'Env_Dino' environment, adds a
    channel dimension, and stacks frames.
2.  PPO Model: Configures a PPO agent with 'CnnPolicy'.
3.  Training: Implements a training loop with checkpointing, a linear
    learning rate schedule, and a custom callback to adjust the entropy
    coefficient (ent_coef) mid-training.
4.  Resuming: Automatically finds and resumes from the latest checkpoint.
5.  Testing: Provides a separate mode to load and test the final trained model,
    and includes a debug feature to save stacked frames as images.

Usage:
    python your_script_name.py       (Resumes training by default)
    python your_script_name.py new   (Starts a new training run)
    python your_script_name.py test  (Tests the final saved model)
"""

import os
import sys
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    CallbackList, 
    CheckpointCallback
)
from PIL import Image 
from env1 import Env_Dino 

TOTAL_TIMESTEPS = 4000000
MODEL_PATH = "models/dino_ppo_final" 
CHECKPOINT_DIR = "models"
CHECKPOINT_FREQ = 20000
FRAME_STACK = 4
ENTROPY_CHANGE_STEP = 500000
ENTROPY_START = 0.01
ENTROPY_END = 0.005
LR_START = 2e-4
LR_END = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChannelFirstWrapper(gym.ObservationWrapper):
    """
    Custom wrapper to re-order observation dimensions.
    
    What it does:
    Converts observations from the (Height, Width) format (e.g., 84, 84)
    to the (Channels, Height, Width) format (e.g., 1, 84, 84).
    
    How it does it:
    It overrides the environment's observation_space to declare the new
    shape. It then uses `np.expand_dims(obs, axis=0)` in the `observation`
    method to add a new dimension at the beginning of the array.
    """
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

def make_env():
    """
    Creates an initialization function for the environment.
    
    What it does:
    Provides a callable function (`_init`) that, when called, creates
    an instance of the `Env_Dino` environment and wraps it with the
    `ChannelFirstWrapper`.
    
    How it does it:
    This factory pattern is required by `DummyVecEnv` so it can
    create environment instances in a controlled way.
    """
    def _init():
        env = Env_Dino()
        env = ChannelFirstWrapper(env)
        return env
    return _init

def create_stacked_env():
    """
    Creates the final vectorized and frame-stacked environment.
    
    What it does:
    Produces the final environment object that will be passed to the PPO model.
    
    How it does it:
    1.  It first creates a vectorized environment using `DummyVecEnv`,
        passing it the `make_env` factory.
    2.  It then wraps the vectorized environment with `VecFrameStack`,
        which concatenates `FRAME_STACK` (e.g., 4) consecutive
        observations along the channel axis. This turns the observation
        shape from (1, 84, 84) into (4, 84, 84), giving the agent
        a perception of motion.
    """
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env


def linear_schedule(initial_value, final_value):
    """
    Returns a scheduler function for the learning rate.
    
    What it does:
    Creates a function that calculates a linearly decaying value.
    
    How it does it:
    The returned function accepts a `progress_remaining` argument (which
    `model.learn()` provides, going from 1.0 down to 0.0) and uses it
    to interpolate between the `initial_value` and `final_value`.
    """
    def func(progress_remaining):
        return final_value + (initial_value - final_value) * progress_remaining
    return func

class EntCoefCallback(BaseCallback):
    """
    A custom callback to change the entropy coefficient (ent_coef) mid-training.
    
    What it does:
    Monitors the total training steps and changes the model's `ent_coef`
    hyperparameter at a predefined step. This allows for high exploration
    at the start and better fine-tuning later.
    
    How it does it:
    Inside the `_on_step` method (called every step), it checks if
    `self.num_timesteps` (the total steps) has exceeded `self.change_step`
    and if the change hasn't already been made. If so, it sets
    `self.model.ent_coef` to the new value and sets a flag to `True`.
    """
    def __init__(self, change_step, new_ent_coef, verbose=0):
        super(EntCoefCallback, self).__init__(verbose)
        self.change_step = change_step
        self.new_ent_coef = new_ent_coef
        self.changed = False

    def _on_step(self) -> bool:
        if not self.changed and self.num_timesteps >= self.change_step:
            if self.verbose > 0:
                print(f"\nCallback: Changing ent_coef to {self.new_ent_coef} at step {self.num_timesteps}")
            self.model.ent_coef = self.new_ent_coef
            self.changed = True
        return True

def train(resume=False):
    """
    Main function to initialize and run the PPO training loop.
    
    What it does:
    Sets up the environment, callbacks, and PPO model. It either loads
    a previous checkpoint or starts a new run, then trains the model
    for `TOTAL_TIMESTEPS` while saving checkpoints.
    
    How it does it:
    1.  Finds the latest checkpoint: If `resume=True`, it scans the
        `CHECKPOINT_DIR` for files matching the prefix and finds the
        one with the highest step number.
    2.  Loads or Creates Model:
        - If resuming, it calls `PPO.load(latest_checkpoint, ...)`. This
          restores the model weights, optimizer state, and all schedule
          progress. It also manually resets the `ent_coef` based on
          the loaded step count.
        - If new, it calls `PPO(...)` with the specified hyperparameters.
    3.  Sets up Callbacks: It creates a `CheckpointCallback` for saving
        and the `EntCoefCallback` for the entropy schedule.
    4.  Trains: It calls `model.learn()` *once* for the full
        `TOTAL_TIMESTEPS` using `reset_num_timesteps=False`. This
        is critical for ensuring the LR schedule and callbacks work
        correctly across resumption.
    5.  Saves: Saves the final model at the end.
    """
    print(f"\nTraining for {TOTAL_TIMESTEPS:,} steps")
    print(f"Frame stack: {FRAME_STACK} frames")
    print(f"Resume: {resume}\n")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    env = create_stacked_env()

    lr_schedule = linear_schedule(LR_START, LR_END)

    latest_checkpoint = None
    if resume:
        checkpoint_files = [
            f for f in os.listdir(CHECKPOINT_DIR) 
            if f.startswith("dino_ppo_checkpoint") and f.endswith(".zip")
        ]
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files, 
                key=lambda f: int(f.split("_")[3])
            )
            latest_checkpoint = os.path.join(CHECKPOINT_DIR, latest_checkpoint)

    if resume and latest_checkpoint:
        print(f"Resuming from {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, device=device)
        
        print(f"Loaded model has {model.num_timesteps} timesteps.")
        if model.num_timesteps < ENTROPY_CHANGE_STEP:
            model.ent_coef = ENTROPY_START
            print(f"Setting ent_coef to {ENTROPY_START} for warmup.")
        else:
            model.ent_coef = ENTROPY_END
            print(f"Setting ent_coef to {ENTROPY_END} for fine-tuning.")
            
    else:
        print("Creating new model (Dino-optimized hyperparameters)")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=4096,
            batch_size=64,
            n_epochs=3,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=ENTROPY_START,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device=device,
            verbose=2
        )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="dino_ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    
    ent_coef_callback = EntCoefCallback(
        change_step=ENTROPY_CHANGE_STEP, 
        new_ent_coef=ENTROPY_END,
        verbose=1
    )
    
    callback_list = CallbackList([checkpoint_callback, ent_coef_callback])
    
    print("\nStarting training...")
    print(f"Checkpoints saved every {CHECKPOINT_FREQ:,} steps to {CHECKPOINT_DIR}/")
    print(f"LR will decay from {LR_START} to {LR_END} over {TOTAL_TIMESTEPS:,} steps.")
    print(f"Ent_coef will change from {ENTROPY_START} to {ENTROPY_END} at {ENTROPY_CHANGE_STEP} steps.\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        progress_bar=True,
        reset_num_timesteps=False
    )
    
    model.save(MODEL_PATH)
    print(f"\n✓ Final model saved to {MODEL_PATH}.zip")
    print(f"✓ Checkpoints saved in {CHECKPOINT_DIR}/")
    
    env.close()

def test(num_episodes=3):
    """
    Loads and tests the final trained PPO model.
    
    What it does:
    Loads the model saved at `MODEL_PATH`, runs it in the environment
    for `num_episodes`, and prints the final score and reward.
    
    How it does it:
    1.  Checks if the model file exists.
    2.  Creates the same stacked environment used for training.
    3.  Loads the model using `PPO.load()`.
    4.  Loops `num_episodes` times:
        - Resets the environment.
        - Enters a `while not done` loop to run the episode.
        - Calls `model.predict(obs, deterministic=True)` to get the
          best action without random exploration.
        - Saves the stacked frames as images for the first 20 steps
          of the first episode for debugging.
        - Calls `env.step(action)` to advance the game.
        - Prints results when the episode finishes.
    5.  Closes the environment.
    """
    model_to_test = MODEL_PATH + ".zip"
    if not os.path.exists(model_to_test):
        print(f"No model found at {model_to_test}. Please train first.")
        return
    
    print(f"\nTesting model: {model_to_test}")
    print(f"Testing for {num_episodes} episodes\n")
    
    save_dir = "_debug_stacked_frames"
    os.makedirs(save_dir, exist_ok=True)
    save_limit = 20
    
    env = create_stacked_env()
    model = PPO.load(model_to_test, env=env, device=device)
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)

            if ep == 0 and steps < save_limit:
                stack = obs[0]
                for i in range(stack.shape[0]):
                    frame_data = stack[i]
                    img = Image.fromarray(frame_data, 'L')
                    img.save(
                        os.path.join(
                            save_dir, 
                            f"step{steps:03d}_frame{i}.png"
                        )
                    )

            obs, reward, done_vec, info = env.step(action)
            
            total_reward += reward[0]
            steps += 1
            done = done_vec[0]
            
            if done:
                score = info[0].get('score', 0)
                print(f"Episode {ep+1}: Score={score}, Reward={total_reward:.2f}, Steps={steps}")
                break
    
    env.close()

if __name__ == "__main__":
    """
    Main entry point for the script.
    
    What it does:
    Parses command-line arguments to decide whether to train or test.
    
    How it does it:
    It checks `sys.argv` (the list of command-line arguments):
    - If the argument is "test", it calls `test()`.
    - If the argument is "new", it calls `train(resume=False)`.
    - Otherwise (no argument or any other argument), it calls
      `train(resume=True)` as the default action.
    """
    
    print(f"Using device: {device}")
    
    print("\n" + "="*70)
    print("IMPORTANT: SERVER MUST BE RUNNING")
    print("="*70)
    print("1. Run: python server.py")
    print("2. Open: http://localhost:8000/index.html")
    print("="*70 + "\n")
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        resume = not (len(sys.argv) > 1 and sys.argv[1] == "new")
        train(resume=resume)