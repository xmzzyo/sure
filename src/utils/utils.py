import os
import time
from contextlib import contextmanager
from functools import wraps
from math import sqrt
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

BASE_DIR, DATA_DIR, RES_DIR = "./", "data/", "saved/"


def set_dirs(is_on_docker, args):
    prefix = "modelarts/user-job-dir/SSD" if is_on_docker else ""
    global BASE_DIR
    BASE_DIR = os.path.join(Path.cwd(), prefix)
    # BASE_DIR = args.train_url
    global DATA_DIR
    DATA_DIR = os.path.join(BASE_DIR, "data/")
    # DATA_DIR = args.data_url
    global RES_DIR
    RES_DIR = os.path.join(BASE_DIR, "saved/")
    # RES_DIR = args.train_url


def list_basic_stat(array):
    return [len(array), np.mean(array) if len(array) > 0 else 0, np.sum(array)]


def traverse_graph(graph, start):
    paths = []

    def helper(path, children):
        if not children:
            paths.append(path[:])
        for i in children:
            path.append(i)
            grandson = graph[i - 1]
            helper(path, grandson)
            path.pop()

    helper([start], graph[start - 1])
    return np.array(paths)


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[1] == W.shape[2]
    scaled_mx = []
    for w in W:
        D = np.diag(np.sum(w, axis=1))
        L = D - w
        lambda_max = eigs(L, k=1, which='LR')[0].real
        scaled = (2 * L) / lambda_max - np.identity(w.shape[0])
        scaled_mx.append(scaled)
    return np.stack(scaled_mx)


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''
    cheb_polynomials_list = []
    for l_tilde in L_tilde:
        N = l_tilde.shape[0]

        cheb_polynomials = [np.identity(N), l_tilde.copy()]

        for i in range(2, K):
            cheb_polynomials.append(2 * l_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        cheb_polynomials_list.append(cheb_polynomials)
    return np.stack(cheb_polynomials_list)


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# and adapted to be synchronous with https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OUNoise:
    def __init__(self, action_dimension, dt=0.01, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.dt = dt
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt \
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# From OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1 - actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = sqrt(np.mean(mean_diff))
    return dist


def graph_data(features, edges, node_num):
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(node_num, node_num), dtype=np.float32)
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    return features, adj


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


@contextmanager
def time_block(label, enable=True):
    if not enable:
        yield
    else:
        start = time.perf_counter()  # time.process_time()
        try:
            yield
        finally:
            end = time.perf_counter()
            print('{} : {} s'.format(label, end - start))


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


def save_rwd(model):
    project_path = Path.cwd()
    if not os.path.exists("saved"):
        os.makedirs("saved")
    save_path = Path(project_path, "saved")
