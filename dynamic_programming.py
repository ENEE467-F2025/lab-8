import gymnasium as gym
import numpy as np


def create_env(map_name="4x4", slippery=True, render=False):
    '''
    Helper function to create Gymnasium environments
    '''
    # Instantiate your environment
    if render:
        render_mode = 'human'
    else:
        render_mode = None

    return gym.make("FrozenLake-v1",
                    map_name=map_name,
                    is_slippery=slippery,
                    render_mode=render_mode)


def compute_action_values(env, state, V, gamma):
    """Helper function to calculate action values for a given state."""
    n_actions = env.action_space.n
    action_values = np.zeros(n_actions)
    
    ########################
    # Your code here
    ########################

    return action_values


def value_iteration(env, max_iterations, gamma, theta):
    """Implementation of Value Iteration."""
    print("VALUE ITERATION")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize value function
    V = np.zeros(n_states)

    # Value Iteration
    
    ########################
    # Your code here
    ########################

    # Derive policy from optimal value function
    policy = np.zeros([n_states, n_actions])
    for s in range(n_states):
        action_values = compute_action_values(env, s, V, gamma)
        best_action = np.argmax(action_values)
        policy[s, best_action] = 1.0

    # Display results
    print("\nOptimal Policy (one-hot):")
    print(policy)

    return policy

def policy_iteration(env, max_iterations, gamma, theta):
    """Implementation of Policy Iteration."""
    print("POLICY ITERATION")
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize policy and value function
    policy = np.ones([n_states, n_actions]) / n_actions  # start with uniform policy
    V = np.zeros(n_states)

    # Policy Iteration

    ########################
    # Your code here
    ########################

    print("\nOptimal Policy (one-hot):")
    print(policy)

    return policy


def play(env, policy, episodes=100):
    """Evaluate a policy for multiple episodes."""
    total_rewards = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(policy[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_rewards += reward
            done = terminated or truncated

    print(f"\nAverage success rate over {episodes} episodes: {total_rewards / episodes:.2f}\n")


if __name__ == '__main__':
    # Create environments for training and evaluation
    map_name = "4x4" # or "8x8"
    slippery = True # `False` for deterministic, `True` for stochastic transitions
    train_env = create_env(map_name, slippery, render=False)
    eval_env = create_env(map_name, slippery, render=True) # set `render` to True to visualize

    # Hyperparameters
    gamma = 0.95  # Discount factor
    theta = 1e-3  # Convergence threshold
    max_iterations = 1000

    alg_names = ["Value Iteration", "Policy Iteration"]
    algs = [value_iteration, policy_iteration]

    for alg_name, alg in zip(alg_names, algs):
        print("*"*50)
        policy = alg(train_env, max_iterations, gamma, theta)

        # Test learned policy
        play(eval_env, policy, episodes=1)
        print("*"*50)
