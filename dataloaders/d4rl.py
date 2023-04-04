import d4rl
import gym

def get_dataloader(**loader_config):
    if loader_config['env'] == 'hopper-medium-replay':
        env = gym.make('hopper-medium-replay-v0')
    elif loader_config['env'] == 'half-cheetah-medium-replay':
        env = gym.make('hopper-medium-replay-v0')
    elif loader_config['env'] == 'walker-medium-replay':
        env = gym.make('walker-medium-replay-v0')
    elif loader_config['env'] == 'reacher-medium-replay':
        env = gym.make('reacher-medium-replay-v0')
    elif loader_config['env'] == 'hopper-medium':
        env = gym.make('hopper-medium-v0')
    elif loader_config['env'] == 'half-cheetah-medium':
        env = gym.make('hopper-medium-v0')
    elif loader_config['env'] == 'walker-medium':
        env = gym.make('walker-medium-v0')
    elif loader_config['env'] == 'reacher-medium':
        env = gym.make('reacher-medium-v0')

    dataset = env.get_dataset()

    # Access the state and action trajectories
    states = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']


    train_data, val_data = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader
