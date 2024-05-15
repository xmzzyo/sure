import os
import pprint
import time
import traceback

import numpy as np
import simpy
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import tianshou.policy
from src.env.env import SSDEnv
from src.env.job import Job
from src.model.actor_critic import Actor, ActorProb
from src.schedule_algo.rl_algo.collector import Collector
from src.schedule_algo.rl_algo.onpolicy import onpolicy_trainer
from src.schedule_algo.rl_algo.reinforce import PGPolicy
from src.utils.config_parser import config
from src.utils.job_reader import CSVReader
from src.utils.logger import logger
from src.utils.utils import DATA_DIR, BASE_DIR, RES_DIR
from tianshou.data import VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, RayVectorEnv
from tianshou.utils import BasicLogger


def run_reinforce(args):
    log_path = os.path.join(RES_DIR, f"{args.model}-reinforce",
                            time.strftime("%m-%d-%H_%M_%S", time.localtime(time.time())))
    args.log_path = log_path
    act_type = args.act_type
    if "discrete" == act_type:
        action_space = Discrete(config.max_task_ins)
    elif "continuous" == act_type:
        action_space = Box(low=-1.0, high=1.0, shape=(1,))

    args.action_shape = action_space.shape or action_space.n

    job_config = np.array(
        CSVReader(os.path.join(DATA_DIR, f"ali_v18-D2000-{args.dataset}.dat")).job_configs)
    job_num = len(job_config)
    if args.job_id:
        jid = args.job_id  # random.randint(0, job_num - 10)
        print("selected job ", jid)
        config.set_config(args, job_config[jid])
        print(config)
        for task_config in job_config[jid].task_configs.values():
            print(task_config)
        Job(simpy.Environment(), job_config[jid]).print_k_hop()
        train_ids = np.array([[jid]] * args.train_num)
        test_ids = np.array([[jid]] * args.test_num)
    else:
        if job_num <= args.train_num:
            train_ids = np.array([*np.arange(job_num), *([np.arange(job_num)] * (args.train_num - job_num))])
        else:
            train_ids = np.array_split(np.arange(job_num), args.train_num)
        test_ids = np.array_split(np.arange(job_num), min(args.test_num, job_num))

    data_info = f"CPU Count: {os.cpu_count()}\nTrain Env/Proc num: {job_num, len(train_ids)}\n" \
                f" Test Env/Proc num: {job_num, len(test_ids)}"
    logger.info(data_info)

    if args.ray:
        import ray
        ray.init(address='auto')
        VecEnv = RayVectorEnv
    else:
        VecEnv = SubprocVectorEnv
    train_envs = VecEnv(
        [lambda jid=(i, x): SSDEnv(args, jid[0], job_config[jid[1]], is_training=True) for i, x in
         enumerate(train_ids)])
    test_envs = VecEnv(
        [lambda jid=(i, x): SSDEnv(args, jid[0], job_config[jid[1]]) for i, x in enumerate(test_ids)])
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    inp_dim, hid_dim = args.inp_dim, args.hid_dim

    from src.model.ssd import BaseNet
    net = BaseNet(inp_dim, hid_dim, args).to(args.device)
    if args.random_sample:
        Agent = tianshou.policy.PGPolicy
    else:
        Agent = PGPolicy

    if "discrete" == act_type:
        actor = Actor(net, args.action_shape, hidden_sizes=[2 * hid_dim, hid_dim],
                      device=args.device, local_rank=args.local_rank).to(args.device)
    else:
        actor = ActorProb(net, args.action_shape, hidden_sizes=[2 * hid_dim, hid_dim], device=args.device,
                          local_rank=args.local_rank).to(args.device)

    # orthogonal initialization
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor.parameters(), lr=args.lr)

    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(
            args.step_per_epoch / args.step_per_collect) * args.epoch

        lr_scheduler = LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    if "discrete" == act_type:
        dist = torch.distributions.Categorical
    else:
        # replace DiagGuassian with Independent(Normal) which is equivalent
        # pass *logits to be consistent with policy.forward
        def dist(*logits):
            return Independent(Normal(*logits), 1)

    policy = Agent(actor, optim, dist,
                   discount_factor=args.gamma,
                   # max_grad_norm=args.max_grad_norm,
                   reward_normalization=args.rew_norm,
                   action_scaling=True,
                   action_bound_method=args.bound_action_method,
                   lr_scheduler=lr_scheduler,
                   action_space=action_space)

    # collector
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # log
    writer = SummaryWriter(log_path)
    basic_logger = BasicLogger(writer, train_interval=1, update_interval=1)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save({
            'model': policy.state_dict(),
            'optim': optim.state_dict(),
        }, os.path.join(log_path, 'checkpoint.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    try:
        result = onpolicy_trainer(
            policy, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.repeat_per_collect, episode_per_test=args.test_num, batch_size=args.batch_size,
            step_per_collect=args.step_per_collect, stop_fn=stop_fn, save_checkpoint_fn=save_checkpoint_fn,
            logger=basic_logger, test_in_train=args.test_in_train, pre_test_episode=args.pre_test_episode)

        pprint.pprint(result)

        args_dict = args.__dict__
        print(args_dict)
        print(dict(config))
        with open(os.path.join(log_path, "config.json"), 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in args_dict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('\n------------------ config ------------------' + '\n')
            for eachArg, value in dict(config).items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    except Exception as e:
        print(traceback.format_exc())
    finally:
        try:
            import moxing as mox
            mox.file.copy_parallel(RES_DIR, 's3://bucket/saved/')
            log_dir = os.path.join(BASE_DIR, "logs")
            if os.path.exists(log_dir):
                mox.file.copy_parallel(log_dir, 's3://bucket/saved/')
        except Exception as e:
            print(f"Copy {RES_DIR} failed.\n Error {e}.")
