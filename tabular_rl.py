''' Implementation of Tabular RL algorithms on Farama-Gymnasium environments
'''

import gymnasium as gym
import numpy as np

# Silence pkg_resources warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# argparse to allow for CLI-based hyperparameter tuning 
import argparse

# plotting and saving utils
import os
import matplotlib
matplotlib.use("Agg")   # use headless backend
import matplotlib.pyplot as plt

# utilities
from utils import (
    summarize_training, summarize_full_run, time_to_first_success,
    summarize_eval, report_run, save_plot, str2bool
)

# script starts
def create_env(env_id, slippery=False, raining=False, render=False):
    '''
    Helper function to create Gymnasium environments
    '''
    # Instantiate your environment
    if render:
        render_mode = 'human'
    else:
        render_mode = None

    if env_id == "FrozenLake-v1":
        return gym.make("FrozenLake-v1",
                        map_name="4x4",
                        is_slippery=slippery,
                        success_rate=4.0/5.0,
                        render_mode=render_mode)
    if env_id == "Taxi-v3":
        return gym.make("Taxi-v3",
                        is_rainy=raining,
                        render_mode=render_mode)
    return gym.make(env_id, render_mode=render_mode)


def q_learning(env, alpha, gamma, epsilon, episodes,  epsilon_decay_rate=0.0001):
    '''
    Implementation of Q-Learning: Tabular Off-Policy RL
    '''
    n_states = env.observation_space.n  # number of states
    n_actions = env.action_space.n  # number of actions

    # Initialize Q-table
    q_table = np.random.random((n_states, n_actions))

    def choose_action(state, epsilon):
        '''
        Implement e-greedy action selection
        '''
        if np.random.rand() < epsilon:
            return env.action_space.sample()  # explore

        return np.argmax(q_table[state])  # exploit

    # Training loop
    ep_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = choose_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Q-learning update (off-policy)
            ########################
            # TODO: Exercise 2 (1)
            ########################

            ########################

            state = next_state
            total_reward += reward
            done = terminated or truncated

        # print(f"Episode: {ep}, Reward = {total_reward}") # print metrics instead and plot evolution
        ep_rewards.append(total_reward)

        # e-decay
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if(epsilon==0):
            alpha = 0.0001

    print("Average reward over last 100 episodes:", np.mean(ep_rewards[-100:]))
    print("Learned Q-table:\n", q_table)

    return ep_rewards, q_table

###########################################
# TODO: Exercise 2 (2a):
# Create a function named sarsa similar to 
# q_learning, but adapt the update logic to 
# use the SARSA update rule
###########################################



###########################################

def play(env, q_table, n_trials=5, seed:int=42):
    '''
    Evaluate the trained agent.
    '''
    ep_rewards = []
    for _ in range(n_trials):
        state, _ = env.reset(seed=seed)
        total_reward = 0
        done = False

        while not done:
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state
            total_reward += reward
            done = terminated or truncated

        ep_rewards.append(total_reward)
    return ep_rewards


if __name__ == '__main__':
    # Create argparse  object
    # Don't change these directly; Modify via CLI calls
    parser = argparse.ArgumentParser(description="A CLI program for Tabular Model-Free RL.")
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='Learning rate'
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.9,
        help='Discount factor'
    )

    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.6,
        help='Exploration factor. Must lie within the open interval: (0, 1)'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=800,
        help='Number of training episodes'
    )

    parser.add_argument(
        '--alg',
        type=str,
        default='qlearn',
        help='Algorithm to use for Tabular RL. Options: `qlearn` and `sarsa`'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Evaluation seed for RNG.'
    )

    parser.add_argument(
        '--slippery',
        type=str2bool,
        default=False,
        help='Whether to use a deterministic (False) or stochastic (True) environment for FrozenLake-v1.'
    )

    parser.add_argument(
        '--raining',
        type=str2bool,
        default=False,
        help='Whether to use a deterministic (False) or stochastic (True) environment for Taxi-v3.'
    )

    parser.add_argument('--mavg_filter_size', type=int, help='Filter size')

    args = parser.parse_args()
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon  
    episodes = args.episodes
    alg = args.alg
    seed = args.seed
    slippery_arg = args.slippery
    raining_arg = args.raining
    mavg_filter_size = args.mavg_filter_size

    # Available training environments
    env_ids = ["FrozenLake-v1", "Taxi-v3", "CliffWalking-v1"]

    #######################
    # TODO: Exercise 2 (2b):
    # Comment this out after filling in 
    # your code for sarsa()
    #######################
    alg_names = {"qlearn": q_learning,
                #  "sarsa": sarsa  
                 }
    ENV_ID = env_ids[0]

    # Create training and testing (evaluation) environments
    train_env = create_env(ENV_ID, slippery=slippery_arg, raining=raining_arg)  # runs headless
    eval_env = create_env(ENV_ID, slippery=slippery_arg, raining=raining_arg, render=False)  # render for evaluation

    # Setup RL hyperparameters
    params = {
        'alpha': alpha,  # learning rate
        'gamma': gamma,  # discount factor
        'epsilon': epsilon,  # exploration factor
        'episodes': episodes,  # training episodes
    }

    if alg not in alg_names:
        raise ValueError(f"Unknown algorithm '{alg}'. Options: {list(alg_names.keys())}")

    try:
        # Training
        ep_rewards, Q_table = alg_names[alg](train_env, **params)
    except Exception as e:
        print(f"An exception has occurred: {e}")
        raise

    # Evaluation
    eval_rewards = play(eval_env, Q_table, n_trials=100, seed=seed)

    # Print metrics
    report_run(f"{alg.upper()} | slippery={slippery_arg}", ep_rewards, eval_rewards=None)

    # save plot
    save_plot(
        ep_rewards,
        alg_name=alg,
        seed=seed,
        params=params,
        slippery=slippery_arg,
        raining=raining_arg,
        mavg_filter_size=mavg_filter_size,
        env_id_arg=ENV_ID
    )
