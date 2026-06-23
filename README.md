# ENEE 467 Fall 2026: Robotics Project Laboratory

## Lab 8: Dynamic Programming and Tabular Reinforcement Learning

This repository contains the instructions well as the necessary code templates for completing the exercises.

## Overview

<p align="left">
    <a href="https://gymnasium.farama.org/" target = "_blank">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium-text.png" width="100px" />
</a>

</p>
Reinforcement learning (RL) studies how an agent selects actions to maximize long-term rewards.
In the tabular setting, states and actions are explicitly enumerated, allowing us to compute value
functions, update policies, and evaluate state–action returns directly. This lab introduces these
core ideas through hands-on implementation of dynamic programming and classical tabular RL
methods, forming the groundwork for later study of continuous-state and deep RL techniques.

## Lab Software
To prepare your system for the RL labs, install the required software using the command-line interface.
Your machine should be running Ubuntu 20.04 or later to ensure compatibility with all libraries
and drivers. If not, consider a virtual machine (VirtualBox, VMware, WSL2) or use the UMIACS
computing service (recommended). We assume teams use UMIACS, though the instructions below should work
natively on any Ubuntu 20.04+ system.

## Accessing the UMIACS Computing Service

If you are enrolled in ENEE 467, you should have received an email confirming your UMIACS class account for accessing the Nexus cluster. The full setup guide is on the [UMIACS Wiki](https://wiki.umiacs.umd.edu/umiacs/index.php/ClassAccounts#Getting_an_account); below we summarize only the essentials. The lab computers already have an SSH client and the GlobalProtect VPN client set up. If you want to connect from your own Ubuntu machine, make sure you have an SSH client installed (e.g., the default `openssh-client`; see [Install OpenSSH](https://documentation.ubuntu.com/server/how-to/security/openssh-server/#install-openssh)), and install the GlobalProtect VPN client following UMD IT’s instructions [here](https://itsupport.umd.edu/itsupport?id=kb_article_view&sysparm_article=KB0016076#mcetoc_1f8qkjlme15).

Start GlobalProtect and connect to the **TunnelAll** gateway. Then confirm that an SSH client is available:
```bash
ssh -V
```
On Ubuntu, this prints the installed OpenSSH version. You can then connect to the Nexus cluster using your UMD Directory ID:
```bash
ssh -Y <umd_id>@nexusclass.umiacs.umd.edu
```

On your first connection, you will be asked to verify the server’s identity. Type `yes` and press `Enter` to store the server’s host key in `~/.ssh/known_hosts`. You will not be prompted again unless the server’s key changes. After confirming the host, enter your **UMD passphrase** when prompted. You will need to authenticate each time you connect or reconnect. Once logged in, complete the rest of the procedure **inside** this remote session.

## RL Environment Setup
Install **Miniconda** and create the environment:
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
exec $SHELL
```
Deactivate the `base` environment and create the RL environment:

```bash
conda deactivate
export CONDA_PLUGINS_AUTO_ACCEPT_TOS="yes"
conda create --name rl467 python=3.11 -y
conda activate rl467
```
Install required packages:
```bash
conda install -c conda-forge swig -y
conda install -c conda-forge matplotlib -y
pip install gymnasium[box2d]
```

## Creating and Editing Files Remotely
We outline two options below:

### A: Vim (terminal editor)
Use `vim` directly on the remote machine:
```bash
vim a.txt
```
Enter **INSERT** mode with <kbd>Shift</kbd>+ <kbd>A</kbd>, edit, then save and quit with `:wq`

### B: VSCode Remote-SSH
Install the **Remote-SSH** extension in VSCode and follow Microsoft’s guide to connect: [Connect to a Remote Host](https://code.visualstudio.com/docs/remote/ssh#_connect-to-a-remote-host), and ensure you use the Nexus SSH string from the UMIACS access section in this README. Once connected, VSCode will show **SSH: nexusclass.umiacs.umd.edu** and allow you to open and edit remote folders. Use <kbd>Ctrl</kbd>+<kbd>`</kbd> to open a remote terminal inside VSCode.

## Test Your Installation
From your Nexus remote session, create a `Labs` folder and clone this repository:

```bash
mkdir ~/Labs && cd ~/Labs
git clone https://github.com/ENEE467-F2025/lab-8.git
```
Then create a new Python file named `test_gym.py` in the `lab-8` repo directory:

```bash
cd ~/Labs/lab-8
vim test_gym.py
```
Copy-paste the following code into `test_gym.py`:

```python
import gymnasium as gym # main library providing sandboxes and RL utilities
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Initialize the environment
env = gym.make("LunarLander-v3", render_mode="human")
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward 
    # and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

Then run the script:

```bash
python3 test_gym.py
```
You should see output similar to: `Action space: Discrete(4) ... Observation space: Box(...)`. Ignore the ALSA driver warnings.

## Lab Instructions
Please follow the [lab manual](Lab_8_RL_Intro.pdf) closely. All instructions are contained inside the lab manual.
