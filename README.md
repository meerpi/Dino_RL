# 🦖 Dino RL: Deep Reinforcement Learning for Chrome Dino

*An attempt to automate the Chrome Dino game using Deep Reinforcement Learning. The main motivation? To finally beat my mobile high score—something I've struggled to replicate on my laptop for some reason.*

This project trains a PPO (Proximal Policy Optimization) agent to play the T-Rex Run game using **TorchRL** and **Playwright**.

## 🚀 Key Features

- **Headless Training:** Uses Playwright to run multiple browser instances in the background without opening visible windows.
- **Parallel Environments:** Trains on 4+ simultaneous game instances for faster convergence.
- **TorchRL Stack:** Built on the modern PyTorch RL library for modularity and speed.
- **Direct Canvas Access:** Captures game frames directly from the browser canvas (no unstable screen capture hacks).

## 🛠️ Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements_torchrl.txt
    playwright install chromium
    ```

2.  **Train the Agent:**
    ```bash
    python train_torchrl.py
    ```
    *Training runs on CUDA if available. Check `models/` for saved checkpoints.*

3.  **Watch it Play:**
    To see the agent in action (visible browser mode):
    ```bash
    python test_playwright.py --model models/dino_torchrl.pth --episodes 5
    ```

## 🧠 Architecture

- **Input:** 4 stacked grayscale frames (84x84)
- **Model:** 3-layer CNN feature extractor + Actor/Critic heads
- **Algorithm:** PPO with GAE (Generalized Advantage Estimation)
