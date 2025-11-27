import gymnasium as gym
import numpy as np

def create_env(env_id, render=False):
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
                        map_name="4x4", # or "8x8"
                        is_slippery=False, # set to `True` for stochastic transitions
                        render_mode=render_mode)
    return gym.make(env_id, render_mode=render_mode)


def q_learning(env, alpha, gamma, epsilon, episodes):
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
            # TODO: Exercise 3 (1)
            ########################

            state = next_state
            total_reward += reward
            done = terminated or truncated

        if ep % 100 == 0:
            print(f"Episode: {ep}, Reward = {total_reward}")
        ep_rewards.append(total_reward)

    print("Average reward over last 100 episodes:", np.mean(ep_rewards[-100:]))

    return q_table


def play(env, q_table, n_trials=5):
    '''
    Evaluate the trained agent.
    '''
    ep_rewards = []
    for _ in range(n_trials):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state
            total_reward += reward
            done = terminated or truncated

        ep_rewards.append(total_reward)


if __name__ == '__main__':
    # Available training environments
    env_ids = ["FrozenLake-v1", "Taxi-v3", "CliffWalking-v0"]
    ENV_ID = env_ids[0]

    # Create training and testing (evaluation) environments
    train_env = create_env(ENV_ID)  # runs headless
    eval_env = create_env(ENV_ID, render=True)  # render for evaluation

    # Setup RL hyperparameters
    params = {
        'alpha': 0.8,  # learning rate
        'gamma': 0.95,  # discount factor
        'epsilon': 0.3,  # exploration factor
        'episodes': 1000,  # training episodes
    }

    Q_table = q_learning(train_env, **params)
    play(eval_env, Q_table, n_trials=100)
