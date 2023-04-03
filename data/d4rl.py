import d4rl
import gym

# Load the dataset
env = gym.make('hopper-medium-replay-v0')
dataset = env.get_dataset()

# Access the state and action trajectories
states = dataset['observations']
actions = dataset['actions']
rewards = dataset['rewards']
terminals = dataset['terminals']