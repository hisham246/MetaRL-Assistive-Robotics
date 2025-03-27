import time
import numpy as np
import torch
from typing import List


class Sampler:
    def __init__(self, dim_features: int, update_func: str = "pick_best", beta_demo: float = 0.1):
        """
        Initializes the sampler.

        :param dim_features: Dimension of feature vectors.
        :param update_func: Options are "rank", "pick_best", and "approx".
        :param beta_demo: Parameter measuring irrationality of human in providing demonstrations.
        """
        self.dim_features = dim_features
        self.update_func = update_func
        self.phi_demos = torch.zeros((1, self.dim_features), dtype=torch.float32)

    def load_demo(self, phi_demos: np.ndarray):
        """
        Loads the demonstrations into the sampler.

        :param phi_demos: A NumPy array containing feature vectors for each demonstration.
                          Has dimension (n_dem x self.dim_features).
        """
        self.phi_demos = torch.tensor(phi_demos, dtype=torch.float32)

    def sample(self, N: int, T: int = 1, burn: int = 1000) -> List:
        """
        Returns N samples from the distribution.

        :param N: Number of samples to draw.
        :param T: If greater than 1, keeps every T-th sample.
        :param burn: Number of initial samples discarded.
        :return: List of samples drawn.
        """
        x = torch.distributions.Uniform(-1.0, 1.0).sample((self.dim_features,))
        
        def f(x):
            return -torch.logsumexp(x, dim=0) + torch.sum(torch.matmul(self.phi_demos, x))

        samples = [x]
        for _ in range(N * T + burn):
            x = torch.distributions.Normal(x, 0.1).sample()
            if f(x) != float('-inf'):
                samples.append(x)

        samples = torch.stack(samples[burn::T])
        samples = samples / (samples.norm(dim=1, keepdim=True) + 1e-8)
        return samples.tolist()


class Sampler_Continuous:
    def __init__(self, dim_features: int, update_func: str = "pick_best", beta_demo: float = 0.1):
        self.dim_features = dim_features
        self.update_func = update_func
        self.phi_demos = torch.zeros((1, self.dim_features), dtype=torch.float32)
        self.previous_samples = None

    def load_demo(self, phi_demos: np.ndarray):
        self.phi_demos = torch.tensor(phi_demos, dtype=torch.float32)

    def f(self, x):
        return -torch.logsumexp(x, dim=0) + torch.sum(torch.matmul(self.phi_demos, x))

    def sample(self, N: int, T: int = 1, burn: int = 1000) -> List:
        x = torch.distributions.Uniform(-1.0, 1.0).sample((self.dim_features,))
        
        if self.previous_samples is not None:
            previous_mean = self.previous_samples.mean(dim=0)
            previous_cov = torch.cov(self.previous_samples.T)
        else:
            previous_mean = torch.zeros(self.dim_features, dtype=torch.float32)
            previous_cov = torch.eye(self.dim_features, dtype=torch.float32) / 5000

        samples = torch.stack([
            torch.distributions.MultivariateNormal(previous_mean, previous_cov).sample()
            for _ in range(N)
        ])
        samples = samples / (samples.norm(dim=1, keepdim=True) + 1e-8)

        self.previous_samples = samples
        return samples.tolist()


class Sampler_Continuous_merge:
    def __init__(self, dim_features: int, alpha: float, update_func: str = "pick_best", beta_demo: float = 0.1):
        self.dim_features = dim_features
        self.update_func = update_func
        self.alpha = alpha
        self.phi_demos = torch.zeros((1, self.dim_features), dtype=torch.float32)
        self.previous_samples = None

    def load_demo(self, phi_demos: np.ndarray):
        self.phi_demos = torch.tensor(phi_demos, dtype=torch.float32)

    def f(self, x):
        return -torch.logsumexp(x, dim=0) + torch.sum(torch.matmul(self.phi_demos, x))

    def sample(self, N: int, T: int = 1, burn: int = 1000) -> List:
        x = torch.distributions.Uniform(-1.0, 1.0).sample((self.dim_features,))
        
        samples = torch.stack([
            torch.distributions.MultivariateNormal(
                torch.zeros(self.dim_features, dtype=torch.float32),
                torch.eye(self.dim_features, dtype=torch.float32) / 5000
            ).sample()
            for _ in range(N)
        ])
        samples = samples / (samples.norm(dim=1, keepdim=True) + 1e-8)

        if self.previous_samples is not None:
            samples = self.alpha * samples + (1 - self.alpha) * self.previous_samples

        self.previous_samples = samples
        return samples.tolist()
