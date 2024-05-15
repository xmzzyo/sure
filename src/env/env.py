import random

import gym
import numpy as np
import simpy
from gym.spaces import Box

from src.env.data_source import DataStream
from src.env.job import Job, k_hop
from src.env.monitor import Monitor
from src.utils.config_parser import config
from src.utils.logger import logger
from tianshou.utils import RunningMeanStd


class SSDEnv(gym.Env):
    def __init__(self, args, i, job_configs, is_training=False, np_state=True):
        logger.info(f"initializing the Env-{i}-{is_training}...")
        # print(f"initializing the Env-{i}-{is_training}...")
        self.args = args
        self.env_id = i
        self.job_configs = job_configs
        self.max_task_ins = config.max_task_ins
        self.max_ins = config.max_ins

        self.lamb1, self.lamb2 = config.lamb1, config.lamb2
        self.obs_span = config.obs_span
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
        self.feat_dim = args.inp_dim
        self._job_adj = None
        self._job_k_nei = None
        self._job_n_nei = None
        self._data_stream = DataStream()
        # self._reset()
        self.sim_end = 0
        self.monitor_ptr = 0
        self.is_training = is_training
        if config.norm_rew and is_training:
            self.reward_rms = RunningMeanStd()
            self.state_rms = RunningMeanStd()
        self.np_state = np_state
        self.rewards = []

    def seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _reset(self):
        self._simpy_env = simpy.Environment()
        self._data_stream.reset(self._simpy_env)
        # Job init
        self.job_config = np.random.choice(self.job_configs)
        self._job = Job(self._simpy_env, self.job_config)
        self._job_adj = None
        self._job_k_nei = None
        self._job_n_nei = None
        self._job.link_from(self._data_stream)
        logger.info(f"initializing job, id: {self.job_id}, task_num: {self._job.task_num}...")
        # Monitor init
        self.monitor = Monitor(self._simpy_env)
        self.monitor.reset(self._job)
        self.monitor_ptr, self.sim_end = 0, 0

        logger.info(f"Reset job {self.job_id}.")

    @property
    def simpy_env(self):
        return self._simpy_env

    @property
    def job(self):
        return self._job

    @property
    def job_id(self):
        return self.job_config.job_index

    @property
    def task_num(self):
        return self.job_config.vertex_num

    @property
    def ins_num(self):
        return self._job.used_ins_num

    @property
    def job_adj(self):
        if self._job_adj is None:
            self._job_adj = np.pad(self._job.adj_matrix,
                                   ((0, config.max_task - self.task_num), (0, config.max_task - self.task_num)))
        return self._job_adj

    @property
    def job_nei(self):
        if self._job_k_nei is None or self._job_n_nei is None:
            self._job_k_nei, self._job_n_nei = k_hop(self._job.adj_matrix, 3, config.max_neis, config.max_task,
                                                     config.pad_task)
        return self._job_k_nei, self._job_n_nei

    def pad_state(self, state):
        return np.pad(state, ((0, 0), (0, config.max_task - self.task_num), (0, 0)))

    def reset(self):
        logger.info(f"Reset SSD Env...")
        self._reset()
        self._simpy_env.process(self.run())
        self._simpy_env.run(until=self.obs_span)
        if self.np_state:
            self.monitor_ptr = len(self.monitor.events)
            next_state = np.stack([event["job_state"] for event in self.monitor.events])
            # zero_vec = np.zeros((1, *next_state.shape[1:]))
            # next_state = np.concatenate([zero_vec, next_state])
            # print(f"reset: {next_state.shape} at {self._simpy_env.now}")
            k_nei, n_nei = self.job_nei
            return {
                "next_state": self.pad_state(next_state),
                "adj_matrix": self.job_adj,
                "task_num": self.task_num,
                "k_nei": k_nei,
                "n_nei": n_nei,
                "workload": self._data_stream.data_log
            }
        else:
            return self._job.metric

    def run(self):
        logger.info(f"start monitor at {self._simpy_env.now}...")
        self._simpy_env.process(self.monitor.run())
        logger.info(f"start data source at {self._simpy_env.now}...")
        self._simpy_env.process(self._data_stream.run())
        logger.info(f"start job {self._job.id} at {self._simpy_env.now}...")
        self._simpy_env.process(self._job.run())
        yield self._simpy_env.timeout(self.obs_span)

    def cal_reward(self, metrics):
        latency = 0.0
        utilized_ratio = 0.0
        for met in metrics:
            latency += -1.0 * np.mean(met["latency"])
            utilized_ratio += np.mean(met["utilized_ratio"])
        rew = metric = np.array([latency / len(metrics), utilized_ratio / len(metrics)])
        # normalize reward
        if config.norm_rew and self.is_training:
            self.reward_rms.update(metric)
            clip_max = 10.0  # this magic number is from openai baselines
            # see baselines/common/vec_env/vec_normalize.py#L10
            rew = (metric - self.reward_rms.mean) / np.sqrt(self.reward_rms.var + 1e-8)
            rew = np.clip(rew, -clip_max, clip_max)  # type: ignore
        if self.args.log_reward and not self.is_training:
            self.rewards.append(metric)
        return self.lamb1 * rew[0] + self.lamb2 * rew[1]

    def clip_action(self, act):
        # act is a ratio, and ranges [-1, 1], not ins num
        # clip action under MAX_INS
        max_task_ins = self.max_task_ins
        desired_ins = self.job.used_ins_num + (act * self.job.used_ins_num).astype(np.int32)
        desired_ins = np.clip(desired_ins, 1, max_task_ins)
        while (self.max_ins - desired_ins.sum()) < 0:
            max_task_ins -= 1
            desired_ins = np.clip(desired_ins, 1, max_task_ins)
        return desired_ins

    # @timethis
    def step(self, act, obs_span=None):
        obs_span = self.obs_span if obs_span is None else obs_span
        if len(act.shape) >= 2:
            act = np.squeeze(act, axis=1)
        if act.size == 1:
            act = [act[0] for _ in range(self.task_num)]
        act = act[:self.task_num]
        act = self.clip_action(act)
        if act.any():
            self._job.take_action(act)
        self.sim_end = self._simpy_env.now + obs_span
        self._simpy_env.run(until=self.sim_end)
        next_state = [event["job_state"] for event in self.monitor.events[self.monitor_ptr:]]
        metrics = [event["metrics"] for event in self.monitor.events[self.monitor_ptr:]]
        self.monitor_ptr = len(self.monitor.events)
        reward = self.cal_reward(metrics)

        if self.np_state:
            next_state = np.stack(next_state)
            k_nei, n_nei = self.job_nei
            next_state = {
                "next_state": self.pad_state(next_state),
                "adj_matrix": self.job_adj,
                "task_num": self.task_num,
                "k_nei": k_nei,
                "n_nei": n_nei,
                "workload": self._data_stream.data_log
            }
        else:
            next_state = self._job.metric
        done = self._simpy_env.now >= config.simpy_runtime
        info = {}
        if done and self.args.log_reward and not self.is_training:
            rwd = np.array(self.rewards)
            rwd = rwd.sum(axis=0)
            info["metrics"] = rwd
            info["reward"] = self.lamb1 * rwd[0] + self.lamb2 * rwd[1]
            self.rewards = []
            logger.info(f"Job {self._job.id} simulation done.")
        return next_state, reward, done, info

    def render(self, mode='human'):
        pass
