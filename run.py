import argparse

import random

import numpy as np
import torch

from src.utils.utils import set_dirs


def common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-num', type=int, default=200)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--job-id', type=int, default=502)
    parser.add_argument('--model', type=str, default='ssd')
    parser.add_argument('--dataset', type=str, default='all')
    parser.add_argument('--trace', type=str, default='ali_v18')
    parser.add_argument('--act-type', type=str, default='continuous')
    parser.add_argument('--inp-dim', type=int, default=27)
    parser.add_argument('--hid-dim', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--kld-w', type=float, default=1e-7)
    parser.add_argument('--nl-w', type=float, default=1e-2)
    parser.add_argument('--mi-w', type=float, default=1e-5)
    parser.add_argument('--lambd', type=float, default=0.7)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    # multiple of env num
    parser.add_argument('--step-per-collect', type=int, default=1000)
    # parser.add_argument('--episode-per-collect', type=int, default=1)
    parser.add_argument('--repeat-per-collect', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--t-k', type=int, default=10)
    parser.add_argument('--s-k', type=float, default=0.5)
    parser.add_argument('--logdir', type=str, default='saved')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--test-in-train', action='store_true')
    parser.add_argument('--pre-test-episode', action='store_true')
    parser.add_argument('--log-reward', action='store_true')
    parser.add_argument('--docker', action='store_true')
    parser.add_argument('--ray', action='store_true')
    parser.add_argument('--local-rank', type=int, default=-1, help='index of current task')
    return parser


def post_args(args):
    # if gpu is to be used
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("=" * 10 + "GPU info" + "=" * 10)
        print("Available devices: ", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
        print("Current device: ", torch.cuda.current_device())
        print("=" * 10 + "GPU info" + "=" * 10)
    print("Using {}".format(args.device))
    if args.local_rank >= 0:
        args = set_distributed(args)
    setup_seed(args.seed)
    set_dirs(args.docker, args)
    return args


def ppo_args(parser=common_args()):
    # ppo special
    parser.add_argument('--use-gat', action='store_true')
    parser.add_argument('--random-sample', action='store_true')
    parser.add_argument('--no-mi', action='store_true')
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.2)
    parser.add_argument('--eps-clip', type=float, default=0.1)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--bound-action-method', type=str, default="clip")
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--reward_threshold', type=float, default=1e10)
    parser.add_argument('--lr-decay', type=int, default=True)
    parser.add_argument('--data_url', default=None, help='Location of data.')
    parser.add_argument('--train_url', default=None, help='Destination of output.')
    args = parser.parse_known_args()[0]
    return post_args(args)


def setup_seed(seed):
    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_distributed(args):
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    return args


if __name__ == "__main__":
    common_parser = common_args()
    cmn_args = common_parser.parse_known_args()[0]
    schedule_algo = cmn_args.model
    print(f"Using {schedule_algo} model.")
    pa = ppo_args(common_parser)
    from src.utils.logger import logger
    from src.schedule_algo.rl_reinforce import run_reinforce

    logger.info(f"Start {schedule_algo} scheduler...")
    run_reinforce(args=pa)
