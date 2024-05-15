from argparse import Namespace

import yaml


def config_parser(config_path):
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return Namespace(**config)


class TestConfig:
    pad_task = 8
    max_task = pad_task + 1
    max_neis = 6
    obs_span = 3
    max_ins = 20
    max_task_ins = 50
    mont_itv = 1
    obs_len = 3

    t_k = 2
    s_k = 3
    time_interval = 0.001


class Config:
    min_task = 10
    pad_task = 46  # 27
    max_task = pad_task + 1
    max_task_ins = 10
    max_ins = 200
    max_neis = 44  # 10  # 44  # 25
    obs_span = 60
    mont_itv = 1
    obs_len = int(obs_span / mont_itv) - 1
    event_len = int((obs_span - 1) / mont_itv)

    t_k = 10
    s_k = 3

    lamb1 = 0.7
    lamb2 = 0.3
    norm_rew = True

    kld_w = 1e-6
    nl_w = 1e-2
    mi_w = 1e-7

    simpy_runtime = 10080
    duration_norm = 1000000.0  # 1000000.0
    time_interval = 0.001
    data_src_itv = 1

    def set_config(self, args, job_config):
        from src.env.job import config2adj, k_hop

        self.pad_task = len(job_config.task_configs)
        self.max_task = self.pad_task + 1
        adj = config2adj(job_config)
        all_K_neis, num_neis = k_hop(adj, 3, self.max_neis, self.max_task, self.pad_task)
        self.max_neis = int(max(num_neis))
        if 0 < args.s_k < 1:
            self.s_k = int(self.max_neis * args.s_k)
        else:
            self.s_k = int(args.s_k)
        self.t_k = args.t_k
        self.kld_w = args.kld_w
        self.nl_w = args.nl_w
        self.mi_w = args.mi_w
        self.lamb1 = args.lambd
        self.lamb2 = 1 - args.lambd

    def __str__(self):
        return f"max_task: {self.max_task}, pad_task: {self.pad_task}, max_neis: {self.max_neis}, t_k: {self.t_k}," \
               f" s_k: {self.s_k}, kld_w: {self.kld_w}, nl_w: {self.nl_w}, mi_w: {self.mi_w}, lambda: {self.lamb1}"

    def keys(self):
        return (
            "min_task", "pad_task", "max_task", "max_task_ins", "max_ins", "max_neis", "obs_span", "mont_itv",
            "obs_len", "kld_w", "nl_w", "mi_w",
            "t_k", "s_k", "lamb1", "lamb2", "norm_rew", "simpy_runtime", "duration_norm", "data_src_itv")

    def __getitem__(self, item):
        return getattr(self, item)


global config
# config = config_parser("src/config.yaml")
config = Config()
