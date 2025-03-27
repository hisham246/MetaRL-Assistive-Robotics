import os
import torch
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import MetaEvaluator, SnapshotConfig
from garage.experiment.task_sampler import ConstructEnvsSampler
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from garage.torch import set_gpu_mode

@wrap_experiment(snapshot_mode='all')
def maml_trainer(ctxt, 
                 env_classes, 
                 seed=1, 
                 epochs=300, 
                 episodes_per_task=40, 
                 meta_batch_size=20):
    set_seed(seed)




class MAMLTrainer:
    def __init__(self, env_classes, seed=1, epochs=300, episodes_per_task=40, meta_batch_size=20):
        self.seed = seed
        self.epochs = epochs
        self.episodes_per_task = episodes_per_task
        self.meta_batch_size = meta_batch_size
        self.env_classes = env_classes
        self.max_episode_length = 100

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # List environments
        envs = [lambda: normalize(GymEnv(env, max_episode_length=self.max_episode_length)) for env in self.env_classes]

        # Set up task sampler
        self.task_sampler = ConstructEnvsSampler(envs)

        # Use the first task to define the policy and value function
        env = envs[0]()

        self.policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )

        self.value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                                       hidden_sizes=(32, 32),
                                                       hidden_nonlinearity=torch.tanh,
                                                       output_nonlinearity=None)

        self.meta_evaluator = MetaEvaluator(test_task_sampler=self.task_sampler, n_test_tasks=2, n_test_episodes=10)

        # Create snapshot config
        snapshot_config = SnapshotConfig(snapshot_dir=os.path.join(os.getcwd(), 'snapshot'),
                                         snapshot_mode='last',
                                         snapshot_gap=1)

        # Create sampler
        self.sampler = LocalSampler(agents=self.policy, envs=env, max_episode_length=self.max_episode_length)

        self.trainer = Trainer(snapshot_config=snapshot_config)

        self.algo = MAMLPPO(env=env,
                            policy=self.policy,
                            task_sampler=self.task_sampler,
                            sampler=self.sampler,
                            value_function=self.value_function,
                            meta_batch_size=self.meta_batch_size,
                            discount=0.99,
                            gae_lambda=1.,
                            inner_lr=0.1,
                            num_grad_updates=1,
                            meta_evaluator=self.meta_evaluator)
        
        # if torch.cuda.is_available():
        #     set_gpu_mode(True)
        # else:
        #     set_gpu_mode(False)
        # self.algo.to()

    def train(self):
        self.trainer.setup(self.algo, self.task_sampler.sample(n_tasks=2))
        self.trainer.train(n_epochs=self.epochs, batch_size=self.episodes_per_task * self.max_episode_length)