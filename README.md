# MarioRL DQN Tutorial

This repository contains code for training and replaying a Deep Q-Network (DQN) agent for the original **Super Mario Bros** environment.

## Recommended Python Version

Python **3.10+** is recommended. The code has been tested with Python 3.12.

## Required Packages

Install the dependencies with `pip`:

```bash
pip install -r requirements.txt
```

`nes_py` must be a version compatible with your Python interpreter. Recent
Python releases (such as 3.12) require **nes_py>=8.2.1**. Older Python
versions may work with previous `nes_py` releases but ensure that the version
matches your environment.

## Running Training

To start training a DQN agent, simply run:

```bash
python train.py
```

Checkpoints and log files will be stored in a time stamped folder under
`checkpoints/`.

## Replaying a Model

After training, you can replay a saved model using:

```bash
python replay.py
```

Modify the `checkpoint` path in `replay.py` if you want to load a specific
saved model.

