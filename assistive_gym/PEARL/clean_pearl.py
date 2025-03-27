from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import ConstructEnvsSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import PEARL
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import (ContextConditionedPolicy,
                                   TanhGaussianMLPPolicy)
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer

@wrap_experiment
def pearl_trainer(ctxt, 
                  env_classes, 
                  seed=1, 
                  num_epochs=500, 
                  num_train_tasks=2, 
                  num_test_tasks=2, 
                  latent_size=5,
                  encoder_hidden_size=200, 
                  net_size=300, 
                  meta_batch_size=16, 
                  num_steps_per_epoch=2000,
                  num_initial_steps=2000, 
                  num_tasks_sample=1, 
                  num_steps_prior=400, 
                  num_extra_rl_steps_posterior=600,
                  batch_size=256, 
                  embedding_batch_size=100, 
                  embedding_mini_batch_size=100, 
                  max_episode_length=200,
                  reward_scale=5., 
                  use_gpu=True):

    set_seed(seed)
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size, encoder_hidden_size)

    # List environments
    envs_list = [lambda: normalize(GymEnv(env, max_episode_length=max_episode_length)) for env in env_classes]

    # Set up task sampler
    task_sampler = ConstructEnvsSampler(envs_list)

    env = task_sampler.sample(n_tasks=num_train_tasks)

    trainer = Trainer(ctxt)

    # Instantiate networks
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    

    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])
    
    vf_env = PEARL.get_env_spec(env[0](), latent_size, 'vf')

    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(env_spec=augmented_env,
                                         hidden_sizes=[net_size, net_size, net_size])
    
    sampler = LocalSampler(agents=inner_policy,
                           envs=env[0](),
                           max_episode_length=max_episode_length,
                           n_workers=1,
                           worker_class=PEARLWorker)
    pearl = PEARL(env=env,
                  policy_class=ContextConditionedPolicy,
                  sampler=sampler,
                  encoder_class=MLPEncoder,
                  inner_policy=inner_policy,
                  qf=qf,
                  vf=vf,
                  num_train_tasks=num_train_tasks,
                  latent_dim=latent_size,
                  encoder_hidden_sizes=encoder_hidden_sizes,
                  test_env_sampler=task_sampler,
                  meta_batch_size=meta_batch_size,
                  num_steps_per_epoch=num_steps_per_epoch,
                  num_initial_steps=num_initial_steps,
                  num_tasks_sample=num_tasks_sample,
                  num_steps_prior=num_steps_prior,
                  num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
                  batch_size=batch_size,
                  embedding_batch_size=embedding_batch_size,
                  embedding_mini_batch_size=embedding_mini_batch_size,
                  reward_scale=reward_scale)

    set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        pearl.to()

    trainer.setup(algo=pearl,
                  env=env)
    
    trainer.train(n_epochs=num_epochs, batch_size=batch_size, store_episodes=True)