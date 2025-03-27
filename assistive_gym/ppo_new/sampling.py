import time
from typing import List, Dict

import numpy as np
import pymc as mc
# import theano as th
# import theano.tensor as tt

import pytensor as pt
import pytensor.tensor as tt


class Sampler(object):
    def __init__(self, dim_features:int, update_func:str="pick_best", beta_demo:float=0.1):
        """
        Initializes the sampler.

        :param n_query: Number of queries.
        :param dim_features: Dimension of feature vectors.
        :param update_func: options are "rank", "pick_best", and "approx". To use "approx", n_query must be 2. Will throw an assertion
            error otherwise.
        :param beta_demo: parameter measuring irrationality of human in providing demonstrations
        :param beta_pref: parameter measuring irrationality of human in selecting preferences
        """
        self.dim_features = dim_features
        self.update_func = update_func


        self.phi_demos = np.zeros((1, self.dim_features))

        self.f = None

    def load_demo(self, phi_demos:np.ndarray):
        """
        Loads the demonstrations into the Sampler.

        :param demos: a Numpy array containing feature vectors for each demonstration.
            Has dimension n_dem x self.dim_features.
        """
        self.phi_demos = phi_demos

    def sample(self, N:int, T:int=1, burn:int=1000) -> List:
        """
        Returns N samples from the distribution defined by applying update_func on the demonstrations and preferences
        observed thus far.

        :param N: number of samples to draw.
        :param T: if greater than 1, all samples except each T^{th} sample are discarded.
        :param burn: how many samples before the chain converges; these initial samples are discarded.
        :return: list of samples drawn.
        """
        x = tt.vector()
        x.tag.test_value = np.random.uniform(-1, 1, self.dim_features)

        start = time.time()
        self.f = pt.function([x], tt.sum(
            [-tt.log(tt.sum(tt.exp(x)))])
                        + tt.sum(tt.dot(self.phi_demos, x)))
        print("Finished constructing sampling function in " + str(time.time() - start) + "seconds")

        # perform sampling
        x = mc.Uniform('x', -np.ones(self.dim_features), np.ones(self.dim_features), value=np.zeros(self.dim_features))
        def sphere(x):
            if (x**2).sum()>=1.:
                return -np.inf
            else:
                return self.f(x)
        p = mc.Potential(
            logp = sphere,
            name = 'sphere',
            parents = {'x': x},
            doc = 'Sphere potential',
            verbose = 0)
        chain = mc.MCMC([x])
        chain.use_step_method(mc.AdaptiveMetropolis, x, delay=burn, cov=np.eye(self.dim_features)/5000)
        chain.sample(N*T+burn, thin=T, burn=burn, verbose=-1)
        samples = x.trace()
        samples = np.array([x/np.linalg.norm(x) for x in samples])

        return samples




class Sampler_Continuous(object):
    def __init__(self, dim_features: int, update_func: str = "pick_best", beta_demo: float = 0.1):
        """
        Initializes the sampler.

        :param dim_features: Dimension of feature vectors.
        :param update_func: Options are "rank", "pick_best", and "approx". To use "approx", n_query must be 2.
        :param beta_demo: Parameter measuring irrationality of human in providing demonstrations.
        """
        self.dim_features = dim_features
        self.update_func = update_func
        self.phi_demos = np.zeros((1, self.dim_features))
        self.f = None
        self.previous_samples = None  # To store the previous samples for refinement

    def load_demo(self, phi_demos: np.ndarray):
        """
        Loads the demonstrations into the Sampler.

        :param phi_demos: a Numpy array containing feature vectors for each demonstration.
        Has dimension n_dem x self.dim_features.
        """
        self.phi_demos = phi_demos

    def sample(self, N: int, T: int = 1, burn: int = 1000) -> List:
        """
        Returns N samples from the distribution defined by applying update_func on the demonstrations and preferences
        observed thus far.

        :param N: Number of samples to draw.
        :param T: If greater than 1, all samples except each T^{th} sample are discarded.
        :param burn: How many samples before the chain converges; these initial samples are discarded.
        :return: List of samples drawn.
        """
        # x = tt.vector()
        # x.tag.test_value = np.random.uniform(-1, 1, self.dim_features)

        # # define update function
        # start = time.time()
        # self.f = pt.function([x], tt.sum(
        #     [-tt.log(tt.sum(tt.exp(x)))]
        #     + tt.sum(tt.dot(self.phi_demos, x))))
        # print("Finished constructing sampling function in " + str(time.time() - start) + " seconds")

        # if self.previous_samples is not None:
        #     previous_mean = np.mean(self.previous_samples, axis=0)
        #     previous_cov = np.cov(self.previous_samples, rowvar=False)
        # else:
        #     previous_mean = np.zeros(self.dim_features)
        #     previous_cov = np.eye(self.dim_features) / 5000

        # x = mc.Uniform('x', -np.ones(self.dim_features), np.ones(self.dim_features), value=previous_mean)

        # Ensure x is a float32 tensor
        x = tt.vector(dtype='float32')  
        x_np = np.random.uniform(-1, 1, self.dim_features).astype(np.float32)
        x.tag.test_value = x_np  # Set test value

        start = time.time()
        # Convert everything to PyTensor tensor before passing it to the function
        self.f = pt.function(
            [x],
            tt.sum(
                [-tt.log(tt.sum(tt.exp(tt.as_tensor_variable(x, dtype='float32'))))]
                + tt.sum(tt.dot(tt.as_tensor_variable(self.phi_demos, dtype='float32'), 
                                tt.as_tensor_variable(x, dtype='float32')))
            )
        )
        print("Finished constructing sampling function in " + str(time.time() - start) + " seconds")

        if self.previous_samples is not None:
            previous_mean = np.mean(self.previous_samples, axis=0).astype(np.float32)
            previous_cov = np.cov(self.previous_samples, rowvar=False).astype(np.float32)
        else:
            previous_mean = np.zeros(self.dim_features, dtype=np.float32)
            previous_cov = np.eye(self.dim_features, dtype=np.float32) / 5000

        # Convert NumPy arrays to lists before using in PyMC3
        x = mc.Uniform(
            'x',
            list(-np.ones(self.dim_features, dtype=np.float32)),
            list(np.ones(self.dim_features, dtype=np.float32)),
            value=list(previous_mean)
        )

        def sphere(x):
            if (x ** 2).sum() >= 1.:
                return -np.inf
            else:
                return self.f(x)

        p = mc.Potential(
            logp=sphere,
            name='sphere',
            parents={'x': x},
            doc='Sphere potential',
            verbose=0)

        # Perform sampling with Adaptive Metropolis
        chain = mc.MCMC([x])
        chain.use_step_method(mc.AdaptiveMetropolis, x, delay=burn, cov=previous_cov)
        chain.sample(N * T + burn, thin=T, burn=burn, verbose=-1)

        # Retrieve samples and normalize them
        samples = x.trace()
        samples = np.array([x / np.linalg.norm(x) for x in samples])

        # Store the new samples for future refinement
        self.previous_samples = samples

        return samples


class Sampler_Continuous_merge(object):
    def __init__(self, dim_features: int, alpha: float, update_func: str = "pick_best", beta_demo: float = 0.1):
        self.dim_features = dim_features
        self.update_func = update_func
        self.phi_demos = np.zeros((1, self.dim_features))
        self.f = None
        self.previous_samples = None
        self.alpha = alpha

    def load_demo(self, phi_demos: np.ndarray):
        """
        Loads the demonstrations into the Sampler.

        :param phi_demos: a Numpy array containing feature vectors for each demonstration.
        Has dimension n_dem x self.dim_features.
        """
        # self.phi_demos = phi_demos
        self.phi_demos = np.array(phi_demos, dtype=np.float32)  # Ensure numpy float32


    def sample(self, N: int, T: int = 1, burn: int = 1000) -> List:
        # x = tt.vector()
        # x.tag.test_value = np.random.uniform(-1, 1, self.dim_features)

        # self.f = pt.function([x], tt.sum(
        #     [-tt.log(tt.sum(tt.exp(x)))]
        #     + tt.sum(tt.dot(self.phi_demos, x))))

        # x = mc.Uniform('x', -np.ones(self.dim_features), np.ones(self.dim_features), value=np.zeros(self.dim_features))

        x = tt.vector(dtype='float32')  # Ensure x is float32
        x_np = np.random.uniform(-1, 1, self.dim_features).astype(np.float32)  # Ensure numpy array
        x.tag.test_value = x_np  # Set test value

        self.f = pt.function(
            [x],
            tt.sum(
                [-tt.log(tt.sum(tt.exp(tt.as_tensor_variable(x, dtype='float32'))))]
                + tt.sum(tt.dot(tt.as_tensor_variable(self.phi_demos, dtype='float32'), 
                                tt.as_tensor_variable(x, dtype='float32')))
            )
        )

        x = mc.Uniform('x', 
               list(-np.ones(self.dim_features, dtype=np.float32)), 
               list(np.ones(self.dim_features, dtype=np.float32)), 
               value=list(np.zeros(self.dim_features, dtype=np.float32)))

        def sphere(x):
            if (x ** 2).sum() >= 1.:
                return -np.inf
            else:
                return self.f(x)

        p = mc.Potential(
            logp=sphere,
            name='sphere',
            parents={'x': x},
            doc='Sphere potential',
            verbose=0)

        chain = mc.MCMC([x])
        chain.use_step_method(mc.AdaptiveMetropolis, x, delay=burn, cov=np.eye(self.dim_features)/5000)
        chain.sample(N * T + burn, thin=T, burn=burn, verbose=-1)

        new_samples = x.trace()
        new_samples = np.array([x / np.linalg.norm(x) for x in new_samples])

        if self.previous_samples is not None:
            combined_samples = self.alpha * new_samples + (1 - self.alpha) * self.previous_samples
        else:
            combined_samples = new_samples

        self.previous_samples = combined_samples

        return combined_samples

