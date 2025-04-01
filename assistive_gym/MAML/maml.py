import torch
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment import MetaEvaluator
from garage.experiment.task_sampler import ConstructEnvsSampler
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

@wrap_experiment(snapshot_mode='all')
def maml_trainer(ctxt, 
                 env_classes, 
                 seed=1, 
                 epochs=200, 
                 episodes_per_task=20, 
                 meta_batch_size=2,
                 max_episode_length=200):
    
    set_seed(seed)

    envs_list = [lambda: normalize(GymEnv(env, max_episode_length=max_episode_length), expected_action_scale=10.0) for env in env_classes]

    task_sampler = ConstructEnvsSampler(envs_list)

    env_instances = [env() for env in envs_list]
    env = MultiEnvWrapper(env_instances, sample_strategy=round_robin_strategy, mode='vanilla')

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler, n_test_tasks=2, n_test_episodes=10)

    trainer = Trainer(ctxt)

    # Networks
    policy = GaussianMLPPolicy(env_spec=env.spec,
                      hidden_sizes=(64, 64),
                      hidden_nonlinearity=torch.tanh,
                      output_nonlinearity=None)
    
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    
    sampler = LocalSampler(agents=policy, envs=env, max_episode_length=max_episode_length)

    maml = MAMLPPO(env=env,
                   policy=policy,
                   task_sampler=task_sampler,
                   sampler=sampler,
                   value_function=value_function,
                   meta_batch_size=meta_batch_size,
                   discount=0.99,
                   gae_lambda=1.,
                   inner_lr=0.1,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator)

    trainer.setup(algo=maml,
                  env=env)

    trainer.train(n_epochs=epochs, 
                  batch_size=episodes_per_task * env.spec.max_episode_length, 
                  store_episodes=True, plot=True)