"""
Utility functions for Tabular RL experiments (FrozenLake, Taxi).
Includes smoothing, statistical summaries, variance estimation, and plotting.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse

# argparse helper
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes','true','t','1'):
        return True
    if v.lower() in ('no','false','f','0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def get_subdir(env_id_arg, slippery, is_raining):
    """
    Determine whether to save results in deterministic or stochastic folder.
    """

    # FrozenLake logic
    if "FrozenLake" in env_id_arg:
        return "stochastic" if slippery else "deterministic"

    # Taxi logic
    if "Taxi" in env_id_arg:
        if is_raining:
            return "stochastic"
        else:
            return "deterministic"

    # default behavior 
    return "stochastic" if slippery else "deterministic"

# Moving Average Smoother
def smooth_curve(x, window):
    """Return moving average of 1D array x using window size
    """
    x = np.asarray(x, dtype=float)
    if window < 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode="same")

# training summary
def summarize_training(ep_rewards, env_id_arg=None, last_n=100):
    """
    Computes reward statistics for the last-N episodes
    
    For FrozenLake:
        reward \in {0,1} --> success_rate = avg reward
    
    For Taxi:
        reward \in integers, success = reward > 0 
        (correct drop-offs give +20)
    """
    ep_rewards = np.asarray(ep_rewards)
    tail = ep_rewards[-last_n:]

    if env_id_arg and "Taxi" in env_id_arg:
        # In Taxiv3, success => positive reward
        success_rate = np.mean(tail > 0)
        avg_last_n = tail.mean()
        var_last_n = tail.var()
        return avg_last_n, var_last_n, success_rate

    else:
        # FrozenLake
        avg_last_n = tail.mean()
        var_last_n = tail.var()
        success_rate = avg_last_n  
        return avg_last_n, var_last_n, success_rate

# full-run stats
def summarize_full_run(ep_rewards):
    ep_rewards = np.asarray(ep_rewards)
    return ep_rewards.mean(), ep_rewards.var()

# time to first success
def time_to_first_success(ep_rewards, env_id_arg=None):
    """
    For Taxi-v3: success = reward > 0
    For FrozenLake: reward == 1
    """
    for i, r in enumerate(ep_rewards):
        if env_id_arg and "Taxi" in env_id_arg:
            if r > 0:
                return i
        else:
            if r == 1:
                return i
    return None

# evaluation summary
def summarize_eval(eval_rewards, env_id_arg=None):
    eval_rewards = np.asarray(eval_rewards)

    if env_id_arg and "Taxi" in env_id_arg:
        success_rate = np.mean(eval_rewards > 0)
        return success_rate, eval_rewards.var()

    else:
        # binary reward
        success_rate = eval_rewards.mean()
        return success_rate, eval_rewards.var()

# report all the things
def report_run(name, ep_rewards, eval_rewards=None, env_id_arg=None, last_n=100):
    """
    Pretty summary for both FrozenLake and Taxi
    """
    ep_rewards = np.asarray(ep_rewards)

    avg_last, var_last, succ = summarize_training(ep_rewards, env_id_arg, last_n)
    avg_full, var_full = summarize_full_run(ep_rewards)
    tfs = time_to_first_success(ep_rewards, env_id_arg)

    parts = name.split(" | ", 1)  
    alg = parts[0].lower() 
    if alg == "qlearn":
        pretty = "Q-learning"
    elif alg == "sarsa":
        pretty = "SARSA"
    else:
        pretty = parts[0]

    if len(parts) == 2:
        name_swap = pretty + " | " + parts[1]
    print("\n" + "-"*65)
    print(f"METRICS SUMMARY: {name_swap}")
    print("-"*65)
    print(f"Full-run avg reward:       {avg_full:.4f}")
    print(f"Full-run variance:         {var_full:.4f}")
    print(f"Last-{last_n} avg reward:       {avg_last:.4f}")
    print(f"Last-{last_n} variance:         {var_last:.4f}")
    print(f"Success rate (last {last_n} eps):  {succ*100:.1f}%")
    print(f"Time to first success:     {tfs}")

    if eval_rewards is not None:
        eval_avg, eval_var = summarize_eval(eval_rewards, env_id_arg)
        print(f"Evaluation success rate:       {eval_avg*100:.1f}%")
        print(f"Evaluation variance:       {eval_var:.4f}")

    print("-"*65 + "\n")

# moving stats
def moving_stats(ep_rewards, window):
    rewards = np.asarray(ep_rewards)
    n = len(rewards)

    means = np.zeros(n)
    stds = np.zeros(n)

    half = window // 2

    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
        window_vals = rewards[left:right]
        means[i] = np.mean(window_vals)
        stds[i] = np.std(window_vals)

    return means, stds

# plot
def save_plot(ep_rewards, alg_name, seed, params,
              slippery, raining,
              mavg_filter_size=None,
              env_id_arg=None,
              last_n=100):

    alpha = params["alpha"]
    gamma = params["gamma"]
    epsilon = params["epsilon"]
    episodes = params["episodes"]

    tex_param_str = (
        rf"$\alpha={alpha}, \gamma={gamma}, \epsilon={epsilon}, "
        rf"\text{{episodes}}={episodes}$"
    )

    # Folder name
    subdir = get_subdir(env_id_arg, slippery, raining)
    out_dir = os.path.join("plots", subdir)
    os.makedirs(out_dir, exist_ok=True)

    filename = (
        f"train_{alg_name}_seed{seed}_"
        f"alpha{alpha}_gamma{gamma}_epsilon{epsilon}_episodes{episodes}.png"
    )
    filepath = os.path.join(out_dir, filename)

    ep_rewards = np.asarray(ep_rewards)
    x = np.arange(len(ep_rewards))

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(6, 4))

    # use moving average filter for FrozenLake
    if env_id_arg == "FrozenLake-v1":
        if mavg_filter_size is None:
            mavg_filter_size = max(10, int(0.10 * episodes))

        means, stds = moving_stats(ep_rewards, mavg_filter_size)
        plt.plot(x, means, linewidth=2, label="Moving Average Reward")

        ymin = np.min(means) - 0.005
        ymax = np.max(means) + 0.005

    # Taxi; no smoothing
    elif env_id_arg == "Taxi-v3":
        plt.plot(x, ep_rewards, linewidth=1.2, label="Reward")

        ymin = np.min(ep_rewards) - 2
        ymax = np.max(ep_rewards) + 2

    else:
        # fallback
        plt.plot(x, ep_rewards, linewidth=1.2)
        ymin = np.min(ep_rewards) - 1
        ymax = np.max(ep_rewards) + 1

    plt.ylim(ymin, ymax)
    plt.xlim(0, len(ep_rewards))

    plt.title(
        (
            f"Reward Evolution ({subdir.upper()}) | "
            f"Last-{last_n} Avg: {ep_rewards[-last_n:].mean():.3f}\n"
            f"{alg_name.upper()} (seed={seed}) | {tex_param_str}"
        ),
        fontsize=10
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.margins(x=0)

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    print(f"Saved reward evolution plot to {filepath}")
