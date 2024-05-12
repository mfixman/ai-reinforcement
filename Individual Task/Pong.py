!pip install gputil
!pip install ray[rllib]
!pip install gym[atari,accept-rom-license]

!pip install -U ipywidgets

import ray
import ray.rllib.algorithms.dqn as dqn
from ray import tune

# Initial Testing
config = dqn.DQNConfig()



config["env"] = "Pong-ram-v4"
config["framework"] = "torch"

#Environment configs
config['model']['grayscale'] = True
config['model']['dim'] = 84
config['model']['zero_mean'] = True
config['sample_batch_size'] = 4
# config['rollout_fragment_length'] = 4

# Rainbow DQN Parameters
config["dueling"] = tune.grid_search([True,False])
config["double_q"] = tune.grid_search([True,False])
config["noisy"] = False

# Rainbow DQN numerical Parameters
config['num_atoms'] = 1
config["n_step"] = 10
config['v_min'] = -12.0
config['v_max'] = 12.0
config['sigma0'] = 0.2

# DQN layers
config["model"] = {"fcnet_hiddens": [512], "fcnet_activation": "relu"}

# Action selection configs
config['exploration_config'] = {'type': 'EpsilonGreedy', 'initial_epsilon':1.0, 'final_epsilon':.001, 'epsilon_timesteps': 40000}

# Replay buffer configurations
# https://docs.ray.io/en/latest/_modules/ray/rllib/utils/replay_buffers/prioritized_replay_buffer.html 
config["replay_buffer_config"] = {
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 75000,
                "prioritized_replay_alpha": 0.7,
                "prioritized_replay_beta": 0.3,
                "prioritized_replay_eps": 0.00001,
                # "replay_sequence_length" : 1
            }

# Training configs
config["train_batch_size"] = 32
config["timesteps_per_iteration"] = 10000   
config["target_network_update_freq"] = 7500
config['gamma'] = 0.99
config['lr'] = 0.001
# config['adam_epsilon'] = 0.0020
config["num_gpus"] = 1
config['num_workers'] = 8
# config['compress_observations'] = True  

analysis3 = tune.run("DQN", name = "Model", metric = "episode_reward_min", num_samples = 1, checkpoint_freq = 10, config = config)


## Testing for learning rate and Update Frequency
config = dqn.DQNConfig()



config["env"] = "Pong-ram-v4"
config["framework"] = "torch"

#Environment configs
config['model']['grayscale'] = True
config['model']['dim'] = 84
config['model']['zero_mean'] = True
config['sample_batch_size'] = 4
# config['rollout_fragment_length'] = 4

# Rainbow DQN Parameters
config["dueling"] = False
config["double_q"] = False
config["noisy"] = False

# Rainbow DQN numerical Parameters
config['num_atoms'] = 1
config["n_step"] = 10
config['v_min'] = -12.0
config['v_max'] = 12.0
config['sigma0'] = 0.2

# DQN layers
config["model"] = {"fcnet_hiddens": [512], "fcnet_activation": "relu"}

# Action selection configs
config['exploration_config'] = {'type': 'EpsilonGreedy', 'initial_epsilon':1.0, 'final_epsilon':.001, 'epsilon_timesteps': 40000}

# Replay buffer configurations
# https://docs.ray.io/en/latest/_modules/ray/rllib/utils/replay_buffers/prioritized_replay_buffer.html 
config["replay_buffer_config"] = {
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 75000,
                "prioritized_replay_alpha": 0.7,
                "prioritized_replay_beta": 0.3,
                "prioritized_replay_eps": 0.00001,
                # "replay_sequence_length" : 1
            }

# Training configs
config["train_batch_size"] = 32
config["timesteps_per_iteration"] = 10000   
config["target_network_update_freq"] = 7500
config['gamma'] = 0.99
config['lr'] = 0.001
# config['adam_epsilon'] = 0.0020
config["num_gpus"] = 1
config['num_workers'] = 8
# config['compress_observations'] = True  

analysis3 = tune.run("DQN", name = "Model", metric = "episode_reward_min", num_samples = 1, checkpoint_freq = 10, config = config)
