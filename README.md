## RL Mario (DQN)

Train a simple Deep Q-Network (DQN) agent to play the first level of Super Mario Bros and then watch it play.

### What you need
- **Python**: 3.10 or newer
- **Windows 10/11 (via WSL2), macOS/Linux (bash)**
- Optional: an **NVIDIA GPU** (PyTorch will auto-use it if available)

---

## Quick start (Windows via WSL2)

Use the WSL2 section below. WSL2 is recommended on Windows for best compatibility.

---

## Quick start (macOS/Linux)

1) Open Terminal and go to the project folder
```bash
cd /path/to/rl_mario
```

2) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Install the requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
# If PyTorch errors on install, try CPU-only wheels:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

4) Train the agent
```bash
python train.py
```

5) Watch Mario play
```bash
python replay.py
```
Use a specific checkpoint:
```bash
export CKPT="/path/to/checkpoints/2025-02-02T16-58-28/mario_net_0.chkpt"
python replay.py
```

---

## Quick start (WSL2 on Windows)

WSL behaves like Linux, so the macOS/Linux steps work. Below are WSL-specific notes.

1) Open Windows Terminal and start your WSL Ubuntu shell, then go to the project folder mounted into WSL (adjust the path to your username):
```bash
cd /mnt/c/Users/kev/OneDrive/Documents/Programming/2-1-2025/rl_mario
```

2) Ensure venv tools and common libraries are installed (first run only):
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential libgl1 libglib2.0-0
```

3) Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

4) Install Python packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
# If PyTorch fails to install GPU wheels, you can use CPU wheels instead:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

5) Train (headless, no window):
```bash
python train.py
```

6) Replay (opens a game window):
```bash
python replay.py
```

If the window does not appear:
- On Windows 11 with WSLg: ensure you are using WSL2 and that other GUI apps open in WSL.
- On Windows 10: install an X server on Windows (e.g., VcXsrv), start it, then in WSL:
```bash
export DISPLAY=$(cat /etc/resolv.conf | awk '/nameserver/ {print $2}'):0.0
export LIBGL_ALWAYS_INDIRECT=1
python replay.py
```

Optional—choose a specific checkpoint:
```bash
export CKPT="/mnt/c/Users/kev/OneDrive/Documents/Programming/2-1-2025/rl_mario/checkpoints/2025-02-02T16-58-28/mario_net_0.chkpt"
python replay.py
```

Verify CUDA inside WSL (optional):
```bash
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## What to expect
- Training is slow to learn at first. Progress plots and logs are saved in `checkpoints/<timestamp>/`.
- A model checkpoint is saved periodically and always once at the end of training.
- Replay opens a game window; the agent plays automatically.

---

## Tips
- Want a checkpoint sooner? In `agent.py`, lower `self.save_every` (e.g., `1e5`).
- Training longer generally produces a better policy.
- GPU is used automatically if available; otherwise CPU is used.

---

## Troubleshooting
- "No checkpoint found" when running replay:
  - Run `python train.py` first, or set `CKPT` to a specific `.chkpt` file.
- PyTorch install errors on Windows:
  - Try the CPU wheels command shown above.
- Replay window does not open:
  - Ensure you ran `replay.py` (training does not display a window), and that you have a desktop session available.
- OverflowError from `nes_py` mentioning `uint8` or NumPy 2.0:
  - Ensure NumPy 1.x is installed (NumPy 2.0 breaks `nes_py`). In your venv run:
    ```bash
    pip install --upgrade "numpy<2.0"
    pip install -r requirements.txt
    ```
  - The warning about Gym being unmaintained is safe to ignore; `gym-super-mario-bros` still depends on Gym.

---

## Memory and performance
- The replay buffer stores up to 10,000 transitions using on-disk memory-mapped storage.
- Typical RAM during training: about 0.5–1.5 GB (more if other apps are open). We recommend **8 GB RAM minimum**, **16 GB** for comfort.
- GPU VRAM usage (if available) is light (~1–2 GB is plenty).


