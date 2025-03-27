from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.task_sampler import ConstructEnvsSampler
from garage.sampler import LocalSampler
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer
import pickle
import os
import re

def get_latest_experiment_dir(base_path='data/local/experiment', prefix='rl2_trainer'):
    candidates = []
    for name in os.listdir(base_path):
        if name == prefix:
            candidates.append((0, name))
        else:
            match = re.fullmatch(f'{prefix}_(\d+)', name)
            if match:
                candidates.append((int(match.group(1)), name))
    if not candidates:
        raise FileNotFoundError(f"No '{prefix}' experiments found in {base_path}")
    
    latest = max(candidates)[1]
    return os.path.join(base_path, latest)

    
@wrap_experiment(snapshot_mode='none')
def rl2_trainer(ctxt,
                env_classes,
                seed=1,
                max_episode_length=2,
                meta_batch_size=2,
                n_epochs=2,
                episode_per_task=4):
    
    set_seed(seed)
    
    with TFTrainer(snapshot_config=ctxt) as trainer:

        envs_list = [lambda env=env: RL2Env(GymEnv(env, max_episode_length=max_episode_length)) for env in env_classes]
        
        task_sampler = ConstructEnvsSampler(envs_list)
        task_updates = task_sampler.sample(meta_batch_size)

        envs = [update() for update in task_updates]
        env_spec = envs[0].spec

        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=64,
                                   env_spec=env_spec,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        sampler = LocalSampler(
            agents=policy,
            envs=task_updates,
            max_episode_length=max_episode_length,
            is_tf_worker=True,
            n_workers=meta_batch_size,
            worker_class=RL2Worker,
            worker_args=dict(n_episodes_per_trial=episode_per_task))

        algo = RL2PPO(meta_batch_size=meta_batch_size,
                      sampler=sampler,
                      task_sampler=task_sampler,
                      env_spec=env_spec,
                      policy=policy,
                      baseline=baseline,
                      episodes_per_trial=episode_per_task,
                      discount=0.99,
                      gae_lambda=0.95,
                      lr_clip_range=0.2,
                      optimizer_args=dict(
                          batch_size=32,
                          max_optimization_epochs=10,
                      ),
                      stop_entropy_gradient=True,
                      entropy_method='max',
                      policy_ent_coeff=0.02,
                      center_adv=False)

        trainer.setup(algo,
                      task_updates)

        trainer.train(n_epochs=n_epochs,
                      batch_size=episode_per_task * max_episode_length * meta_batch_size,
                      store_episodes=True)

        # Save policy
        params = policy.__getstate__()
        target_dir = get_latest_experiment_dir()
        os.makedirs(target_dir, exist_ok=True)

        with open(os.path.join(target_dir, 'rl2_policy.pkl'), 'wb') as f:
            pickle.dump(params, f)