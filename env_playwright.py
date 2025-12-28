"""
This module provides a robust, Playwright-based interface for the Chrome Dino game.
It allows you to train Reinforcement Learning agents in a headless environment, 
enabling parallel training without the need for visual rendering or WebSocket bottlenecks.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image
import io
import os
import time
from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext


class DinoPlaywrightEnv(gym.Env):
    """
    A custom Gymnasium environment for the Chrome Dino game, controllable via Playwright.

    I chose Playwright over Selenium because it supports 'browser contexts', which are like 
    incognito windows that share a single browser process. This means we can run 4+ environments 
    in parallel without eating up all the RAM!
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        game_url: str = None,
        headless: bool = True,
        render_mode: str = None,
        frame_skip: int = 4,
        observation_size: tuple = (84, 84),
    ):
        """
        Initializes the environment.

        Args:
            game_url: Path to the game file. If not provided, it finds the local 'index.html'.
            headless: We usually run headless (no window) to speed up training, but can set this 
                     to False if we want to watch the Dino jump around.
            render_mode: 'human' for display, 'rgb_array' for the raw pixels the agent sees.
            frame_skip: The agent only sees every 4th frame. This is a standard RL trick (like in Atari)
                       because nothing much changes in 1/60th of a second, so it saves computation.
            observation_size: We downscale the screen to 84x84 grayscale to make it easier for 
                            the CNN to process.
        """
        super().__init__()
        
        self.headless = headless
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.observation_size = observation_size
        
        if game_url is None:
            game_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 't-rex-runner')
            self.game_url = f"file://{game_dir}/index.html"
        else:
            self.game_url = game_url
        
        self.action_space = spaces.Discrete(3)  # 0: None, 1: Duck, 2: Jump
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=observation_size,
            dtype=np.uint8
        )
        
        self._playwright = None
        self._browser: Browser = None
        self._context: BrowserContext = None
        self._page: Page = None
        
        self.current_step = 0
        self.max_steps = 10000
        self.last_score = 0
        self.episode_reward = 0
        self.episode_count = 0
        
        self._init_browser()
    
    def _init_browser(self):
        """Starts the Playwright engine and launches the Chromium browser."""
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-gpu',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-web-security',
            ]
        )
        self._create_page()
    
    def _create_page(self):
        """Creates a new isolated browser context and loads the game page."""
        if self._context:
            self._context.close()
        
        self._context = self._browser.new_context(
            viewport={'width': 600, 'height': 150}
        )
        self._page = self._context.new_page()
        self._page.goto(self.game_url)
        self._page.wait_for_load_state('networkidle')
        self._page.wait_for_selector('canvas', timeout=5000)
        time.sleep(0.2)
    
    def _get_game_state(self) -> dict:
        """
        Injects JavaScript into the browser to steal the internal game state.
        
        This is a bit of a hack, but it allows us to get the exact score, speed, and 
        obstacle positions directly from the game's variables (`Runner.instance_`).
        Much cleaner than trying to OCR the score from the screen!
        """
        state = self._page.evaluate("""
            () => {
                const runner = Runner.instance_;
                if (!runner) return null;
                
                // We need to know where the obstacles are to help debug, 
                // though the agent mainly looks at the pixels.
                let obstacles = [];
                if (runner.horizon && runner.horizon.obstacles) {
                    for (let obs of runner.horizon.obstacles) {
                        obstacles.push({
                            type: obs.typeConfig.type,
                            x_pos: obs.xPos,
                            y_pos: obs.yPos,
                            width: obs.width,
                            height: obs.typeConfig.height,
                            distance: obs.xPos - runner.tRex.xPos
                        });
                    }
                }
                
                return {
                    crashed: runner.crashed,
                    playing: runner.playing,
                    score: Math.ceil(runner.distanceRan),
                    speed: runner.currentSpeed,
                    activated: runner.activated,
                    obstacles: obstacles,
                    trex: {
                        x_pos: runner.tRex.xPos,
                        y_pos: runner.tRex.yPos,
                        jumping: runner.tRex.jumping,
                        ducking: runner.tRex.ducking,
                        velocity: runner.tRex.jumpVelocity
                    }
                };
            }
        """)
        return state
    
    def _get_observation(self) -> np.ndarray:
        """
        Takes a screenshot of the browser canvas and processes it for the Neural Network.
        
        We do a few image processing steps here:
        1. Capture raw bytes from Playwright
        2. Convert to grayscale (color doesn't matter for Dino)
        3. Invert colors (we want white features on black background, typically easier for CNNs)
        4. Resize to 84x84 (standard size from the DeepMind Atari papers)
        """
        canvas = self._page.locator('canvas')
        screenshot_bytes = canvas.screenshot()
        
        img = Image.open(io.BytesIO(screenshot_bytes))
        img = img.convert('L')
        img = Image.eval(img, lambda x: 255 - x)
        img = img.resize(self.observation_size, Image.BICUBIC)
        
        return np.array(img, dtype=np.uint8)
    
    def _start_game(self):
        """Simulates a spacebar press to start or restart the game."""
        self._page.keyboard.press('Space')
        time.sleep(0.1)
    
    def _do_action(self, action: int):
        """
        Performs the specified action in the browser.
        
        Args:
            action: 0 (Do nothing), 1 (Duck), or 2 (Jump).
        """
        if action == 1:
            self._page.keyboard.down('ArrowDown')
            time.sleep(0.05)
            self._page.keyboard.up('ArrowDown')
        elif action == 2:
            self._page.keyboard.press('ArrowUp')
    
    def _calculate_reward(self, state: dict, action: int) -> float:
        """
        Calculates the reward to guide the agent.
        
        This was tricky to tune!
        - **Survival Reward (+0.1):** We want it to stay alive.
        - **Score Reward (+ difference):** If the score goes up, give it a treat.
        - **Death Penalty (-10.0):** Crashing is very bad.
        - **Action Penalty (-0.01):** We subtract a tiny bit for jumping or ducking. 
          Why? To stop it from spamming jump constantly (it looks silly and is inefficient). 
          We want it to only jump when it *needs* to.
        """
        if state['crashed']:
            return -10.0
        
        reward = 0.1
        
        score = state.get('score', 0)
        if score > self.last_score:
            reward += (score - self.last_score) * 0.01
        self.last_score = score
        
        if action in [1, 2]:
            reward -= 0.01
        
        return reward
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state for a new episode.
        
        Returns:
            The initial observation and info dictionary.
        """
        super().reset(seed=seed)
        
        if self.current_step > 0:
            print(f"\n{'='*50}")
            print(f"Episode {self.episode_count} Summary:")
            print(f"  Steps: {self.current_step}")
            print(f"  Score: {self.last_score}")
            print(f"  Reward: {self.episode_reward:.2f}")
            print(f"{'='*50}\n")
        
        self.episode_count += 1
        self.current_step = 0
        self.last_score = 0
        self.episode_reward = 0
        
        # Browser operations might fail if the connection is lost (e.g. manual closure)
        # We let these specific errors propagate or handle them if needed, 
        # but avoid wrapping everything in a generic try/except.
        self._page.reload()
        self._page.wait_for_load_state('networkidle')
        self._page.wait_for_selector('canvas', timeout=5000)
        time.sleep(0.2)
        
        self._start_game()
        time.sleep(0.1)
        
        obs = self._get_observation()
        state = self._get_game_state()
        
        info = {
            'score': state.get('score', 0) if state else 0,
            'step': 0
        }
        
        return obs, info
    
    def step(self, action: int):
        """
        Runs one timestep of the game.
        
        This logic is wrapped in a try/except merely to catch things like the 
        browser crashing or the network hanging. If that happens, we just tell 
        the RL loop that the episode ended unexpectedly.
        """
        self.current_step += 1
        action = int(action)
        
        try:
            self._do_action(action)
            # Sleep here to simulate the frame rate (60 FPS / frame_skip)
            time.sleep(0.016 * self.frame_skip)
            
            state = self._get_game_state()
            
            if state is None:
                obs = np.zeros(self.observation_size, dtype=np.uint8)
                return obs, -10.0, True, False, {'error': 'game_not_ready'}
            
            obs = self._get_observation()
            reward = self._calculate_reward(state, action)
            self.episode_reward += reward
            
            terminated = state['crashed']
            truncated = self.current_step >= self.max_steps
            
            info = {
                'score': state.get('score', 0),
                'step': self.current_step,
                'speed': state.get('speed', 0)
            }
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            # If the browser disconnects or something breaks, fail the episode gracefully
            obs = np.zeros(self.observation_size, dtype=np.uint8)
            return obs, -10.0, True, False, {'error': str(e)}
    
    def render(self):
        """Returns the current observation frame."""
        if self.render_mode == "rgb_array":
            return self._get_observation()
    
    def close(self):
        """Releases the browser resources."""
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()


def make_env(env_id: int = 0, headless: bool = True, game_url: str = None):
    """
    Helper function to create environment instances, primarily for Multi ENV environments.
    """
    def _init():
        env = DinoPlaywrightEnv(
            game_url=game_url,
            headless=headless,
        )
        return env
    return _init


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visible', action='store_true', help='Show browser window')
    args = parser.parse_args()
    
    headless = not args.visible
    print(f"Testing DinoPlaywrightEnv (headless={headless})...")
    
    env = DinoPlaywrightEnv(headless=headless)
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 20 == 0:
            print(f"Step {i}: reward={reward:.3f}, score={info.get('score', 0)}, terminated={terminated}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            obs, info = env.reset()
    
    env.close()
    print("Test complete!")
