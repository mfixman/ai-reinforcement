import torch
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from itertools import product
from torch.nn import MaxPool2d
import pprint
import logging

Models = dict(
    small = dict(
        fcnet_hiddens = [64, 64],
        fcnet_activation = 'relu',
    ),
    large = dict(
        fcnet_hiddens = [250, 256, 128],
        fcnet_activation = 'relu',
    ),
    cnn = dict(
        conv_filters = [
            [32, [8, 8], 4],
            [64, [4, 4], 4],
            [64, [2, 2], 6],
        ],
        fcnet_hiddens = [512],
        fcnet_activation = 'relu', 
    )
)

def get_ppo_model(lr, gamma, model):
    config = PPOConfig().environment(
        "ALE/SpaceInvaders-v5"
    ).resources(
        num_gpus = 1,
    ).framework(
        "torch"
    ).rollouts(
        num_rollout_workers = 10,
        num_envs_per_worker = 5,
        observation_filter = 'NoFilter',
        rollout_fragment_length = 100,
    ).training(
        gamma = gamma,
        lambda_ = 0.95,
        kl_coeff = 0.5,
        clip_param = 0.1,
        entropy_coeff = 0.01,
        train_batch_size = 5000,
        sgd_minibatch_size = 500,
        num_sgd_iter = 10,
        vf_clip_param = 10.0,
        vf_share_layers = True
    ).evaluation(
        evaluation_interval = 1,
        evaluation_duration = 10,
        evaluation_duration_unit = 'episodes'
    )
    config.batch_mode = 'truncate_episodes'

    return config.build()

def train_epoch(ppo):
    results = ppo.train()
    results['policy_loss'] = results['info']['learner']['default_policy']['learner_stats']['policy_loss']
    results['vf_loss'] = results['info']['learner']['default_policy']['learner_stats']['vf_loss']
    results['entropy'] = results['info']['learner']['default_policy']['learner_stats']['entropy']
    return results

def main():
    ray.init()

    logging.basicConfig(
        level = logging.INFO,
        format = '[%(asctime)s] %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S',
    )

    lrs = [.001]
    gammas = [.99]
    model_names = ['cnn']

    params = ['lr', 'gamma', 'model_name']
    training = ['episode']
    metrics = ['episode_reward_mean', 'episode_len_mean', 'policy_loss', 'vf_loss', 'entropy']
    print(','.join(params + training + metrics), flush = True)
    for values in product(lrs, gammas, model_names):
        lr, gamma, model_name = values
        model = get_ppo_model(lr, gamma, Models[model_name])

        for episode in range(1, 2001):
            if episode % 10 == 1:
                logging.info(f'Starting episode {episode}')

            if episode % 100 == 1:
                path = model.save(checkpoint_dir = f'./checkpoints/ep{episode}')
                logging.info(f'Saved checkpoint at {path}')

            pert = [episode]

            results = train_epoch(model)
            my_results = [results[m] for m in metrics]
            print(','.join(str(x) for x in (list(values) + pert + my_results)), flush = True)

if __name__ == '__main__':
    main()
