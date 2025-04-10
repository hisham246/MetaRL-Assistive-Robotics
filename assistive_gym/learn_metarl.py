import numpy as np
import torch
import gym
import importlib

# tracing experiments
from dowel import tabular

import hydra
from omegaconf import DictConfig

# Meta-Learning
from assistive_gym.MAML.maml import maml_trainer
from assistive_gym.PEARL.pearl import pearl_trainer
from assistive_gym.RL2.rl2 import rl2_trainer

def make_env(args, env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        # environments are modified, if you with to further modify, please go to assistive_gym.envs
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(args.seed)

    return env
        

def train_maml(args: DictConfig):
    # Define your environment classes
    env_classes = args.envs.split(',')

    # Create environments using make_env
    envs = [make_env(args, env_name, coop=('Human' in env_name)) for env_name in env_classes]

    # Initialize MAML
    maml_trainer(
        env_classes=envs,
        seed=args.seed,
        epochs=args.MAML.epochs,
        episodes_per_task=args.MAML.episodes_per_task,
        meta_batch_size=args.MAML.meta_batch_size
    )

def train_pearl(args: DictConfig):
    # Define your environment classes
    env_classes = args.envs.split(',')
    envs = [make_env(args, env_name, coop=('Human' in env_name)) for env_name in env_classes]

    # Initialize PEARL
    pearl_trainer(env_classes=envs,
                  seed=args.seed,
                  num_epochs=args.PEARL.epochs,
                  num_train_tasks=args.PEARL.num_train_tasks,
                  num_test_tasks=args.PEARL.num_test_tasks,
                  latent_size=args.PEARL.latent_size,
                  encoder_hidden_size=args.PEARL.encoder_hidden_size,
                  net_size=args.PEARL.net_size,
                  meta_batch_size=args.PEARL.meta_batch_size,
                  num_steps_per_epoch=args.PEARL.num_steps_per_epoch,
                  num_initial_steps=args.PEARL.num_initial_steps,
                  num_tasks_sample=args.PEARL.num_tasks_sample,
                  num_steps_prior=args.PEARL.num_steps_prior,
                  num_extra_rl_steps_posterior=args.PEARL.num_extra_rl_steps_posterior,
                  batch_size=args.PEARL.pearl_batch_size,
                  embedding_batch_size=args.PEARL.embedding_batch_size,
                  embedding_mini_batch_size=args.PEARL.embedding_mini_batch_size,
                  max_episode_length=args.PEARL.max_episode_length,
                  reward_scale=args.PEARL.reward_scale,
                  use_gpu=args.PEARL.use_gpu)

def train_rl2(args: DictConfig):
    # Define your environment classes
    env_classes = args.envs.split(',')
    envs = [make_env(args, env_name, coop=('Human' in env_name)) for env_name in env_classes]

    # Initialize RL2
    rl2_trainer(env_classes=envs,
                seed=args.seed,
                max_episode_length=args.RL2.max_episode_length,
                meta_batch_size=args.RL2.meta_batch_size,
                n_epochs=args.RL2.n_epochs,
                episode_per_task=args.RL2.episode_per_task)

@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(args: DictConfig):
    print("Running experiment:", args.algo)
    print("Environments:", args.envs.split(','))

    if args.algo == 'MAML':
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        train_maml(args)

    elif args.algo == 'PEARL':
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        train_pearl(args)

    elif args.algo == 'RL2':
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        train_rl2(args)

if __name__ == "__main__":
    main()