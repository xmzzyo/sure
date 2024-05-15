import math
import random

import numpy as np
import simpy
from numba import njit

from src.env.config import TaskInstanceConfig
from src.utils.config_parser import config
from src.utils.logger import logger
from src.utils.utils import list_basic_stat

# time cost of restarting job
RESTART_COST = 1
LATENCY_MAX = config.mont_itv


class Batch:
    """
    Data batch
    """

    def __init__(self, bid, inp_time, pro_time=0, size=0):
        self.bid = bid
        self.size = size
        self.inp_time = inp_time
        self.pro_time = pro_time
        self.out_time = None

    def __str__(self):
        return f"bid: {self.bid}\tsize: {self.size}"


class BroadcastPipe:
    """A Broadcast pipe that allows one process to send messages to many.

    This construct is useful when message consumers are running at
    different rates than message generators and provides an event
    buffering to the consuming processes.

    The parameters are used to create a new
    :class:`~simpy.resources.store.Store` instance each time
    :meth:`get_output_conn()` is called.

    """

    def __init__(self, simpy_env, capacity=simpy.core.Infinity):
        self.simpy_env = simpy_env
        self.capacity = capacity
        self.pipes = []

    def put(self, value, pids=None):
        """Broadcast a *value* to all receivers."""
        if not self.pipes:
            raise RuntimeError('There are no output pipes.')
        if pids is None:
            events = [store.put(value) for store in self.pipes]
        else:
            events = [self.put_one(value, pid) for pid in pids]
        return self.simpy_env.all_of(events)  # Condition event for all "events"

    def put_one(self, value, pid):
        return self.pipes[pid].put(value)

    def get_output_conn(self):
        """Get a new output connection for this broadcast pipe.

        The return value is a :class:`~simpy.resources.store.Store`.

        """
        pipe = simpy.Store(self.simpy_env, capacity=self.capacity)
        self.pipes.append(pipe)
        return pipe

    def reset(self):
        self.pipes = []

    @property
    def num_rev(self):
        return len(self.pipes)


class TaskInstance:
    """
    in_pipe: the pipe receiving message from other tasks
    out_pipe: the pipe broadcasting messages to other tasks
    """

    def __init__(self, simpy_env, task, task_instance_index, task_instance_config, in_pipe, out_pipe):
        self.simpy_env = simpy_env
        self.task = task
        self.task_instance_index = task_instance_index
        self.config = task_instance_config
        self.cpu = task_instance_config.cpu
        self.duration = task_instance_config.duration

        self.machine = None
        self.process = None
        self.new = True

        self.started = False
        self.finished = False
        self.is_free = True
        self.started_timestamp = None
        self.finished_timestamp = None

        self.completed_batch = set()
        self.queued_batch = []

        self.in_pipe = in_pipe
        self.out_pipe = out_pipe

    @property
    def id(self):
        return str(self.task.id) + '-' + str(self.task_instance_index)

    def run(self):
        """
        run task instance, waiting data from upstream tasks, return message to Task
        :return:
        """
        # self.cluster.waiting_tasks.remove(self)
        # self.cluster.running_tasks.append(self)
        # self.machine.run(self)
        while True:
            try:
                msg = yield self.in_pipe.get()
                self.is_free = False
                batch = msg["data"]
                self.queued_batch.append(batch)
                # logger.info(f"task ins {self.id} executes {batch.bid}-{batch.size}")
                yield self.simpy_env.timeout(math.ceil(batch.size) * self.duration)
                # logger.info(f"Task-Ins {self.id} done Batch {batch.bid} out at {self.simpy_env.now}")
                batch.out_time = self.simpy_env.now
                self.completed_batch.add(batch.bid)
                self.queued_batch.pop()
                done_msg = {"time": self.simpy_env.now, "type": "ins_done",
                            "data": (self.task_instance_index, batch.bid)}
                self.out_pipe.put(done_msg)
                self.is_free = True
            except simpy.Interrupt:
                break
        self.completed_batch = set()
        self.queued_batch = []
        self.started = False
        # logger.info(f"Task-Ins {self.id} stop at {self.simpy_env.now}.")

    def schedule(self, machine=None):
        self.started = True
        self.started_timestamp = self.simpy_env.now
        # self.machine = machine
        # self.machine.run_task_instance(self)
        self.process = self.simpy_env.process(self.run())
        # logger.info(f"Task-Ins {self.id} start at {self.simpy_env.now}.")


class Task:
    def __init__(self, simpy_env, job, task_config, in_pipe=None, out_pipe=None):
        self.simpy_env = simpy_env
        self.job = job
        self.task_index = task_config.task_index
        self.task_config = task_config
        self._ready = False
        self._run = False
        self._parents = None
        self._children = None
        self.is_source = task_config.is_source
        self.is_sink = task_config.is_sink
        self.sink_batch = []
        self.exe_batch = {}

        self.in_pipe = in_pipe
        self.out_pipe = out_pipe
        self.instance_sender_pipe = BroadcastPipe(simpy_env)
        self.instance_receiver_pipe = simpy.Store(simpy_env)

        self.task_instance_config = TaskInstanceConfig(task_config)
        self._build_ins(int(self.task_config.parallelism))
        self.queued_batch_pipe = simpy.Store(simpy_env)
        self.complete_batch = set()
        self.wrk_ins = dict()
        self.stop = False
        self.prev_window_latency = None
        # self.start_task_instance(None)

    def fetch_batch(self):
        """
        monitoring message from other tasks, if message is from parent task, execute batch
        :return:
        """
        while True:
            msg = yield self.in_pipe.get()
            # "stop" signal
            if "stop" == msg["type"]:
                # print(f"stop - {self.id}, now is ", time.time())
                self.stop = True
                yield self.simpy_env.process(self.restart(msg["action"]))
                self.stop = False
                # with self.exe_idx to construct a large batch
                for bid, batch in self.exe_batch.items():
                    if bid not in self.complete_batch:
                        self.assign_batch(batch)
                    else:
                        self.exe_batch.pop(bid)
                continue
            else:
                # data signal
                batch = msg["batch"]
            if batch.bid in self.complete_batch:
                continue
            if self.is_source:
                pass
            else:
                done_task = msg["task"]
                # check if passed from predecessor tasks
                if done_task not in self.task_config.parent_indices:
                    continue
                # check if al predecessor tasks have done this batch
                if not self.ready(batch.bid):
                    continue
            self.assign_batch(batch)
            self.exe_batch[batch.bid] = batch

    def execute_batch(self):
        while True:
            while True:
                ins_done_msg = yield self.instance_receiver_pipe.get()
                bid = ins_done_msg["data"][1]
                if bid not in self.complete_batch and self.ins_done_batch(bid):
                    self.complete_batch.add(bid)
                    break
            batch = self.batch_by_id(bid)
            self.done_info.append({"time": self.simpy_env.now,
                                   "batch_id": bid,
                                   "batch_size": batch.size,
                                   "batch_latency": self.simpy_env.now - batch.pro_time,
                                   # "batch_proc_time":
                                   })
            self.exe_batch.pop(bid)
            # logger.info(
            #     f"Task {self.id} done Batch {batch.bid}-{batch.size} out at {self.simpy_env.now}")
            # pass batch down to children
            if not self.is_sink:
                batch.pro_time = self.simpy_env.now
                msg = {"time": self.simpy_env.now, "type": "down_data", "task": self.task_index,
                       "batch": batch}
                self.out_pipe.put(msg)
            else:
                batch.out_time = self.simpy_env.now
                logger.info(f"Batch {bid} sink at {self.simpy_env.now}")
                self.sink_batch.append(batch)

    def assign_batch(self, batch):
        # need slot_num process to execute batch
        if batch.size < self.parallelism:
            slot_num = batch.size
        else:
            slot_num = self.parallelism
        free_slot = []
        for ins in self.task_instances:
            if ins.started and ins.is_free:
                free_slot.append(ins.task_instance_index)
                if len(free_slot) == slot_num:
                    break
        if len(free_slot) < slot_num:
            extra_slot = random.sample(set(range(self.parallelism)) - set(free_slot), slot_num - len(free_slot))
            free_slot.extend(extra_slot)
        self.wrk_ins[batch.bid] = free_slot
        base_bsz = batch.size // slot_num
        assign_bsz = np.full(slot_num, base_bsz)
        # assign_bsz[np.random.choice(np.arange(slot_num), batch.size - base_bsz * slot_num)] += 1
        assign_bsz[:(batch.size - base_bsz * slot_num)] += 1
        # print(f"Assign {self.id} batch {batch.bid}-{batch.size}-{slot_num} Ins {free_slot}, bsz {assign_bsz}")
        for slot_id, bsz in zip(free_slot, assign_bsz):
            msg = {"time": self.simpy_env.now, "type": 'ins_batch',
                   "data": Batch(batch.bid, batch.inp_time, batch.pro_time, bsz)}
            self.instance_sender_pipe.put_one(msg, slot_id)
        # logger.info(f"Task {self.id} execute batch {batch.bid}-{batch.size} at {self.simpy_env.now}")

    def run(self):
        self.simpy_env.process(self.fetch_batch())
        self.simpy_env.process(self.execute_batch())

    def start_task_instance(self, machine=None):
        for task in self.task_instances:
            task.schedule()
        # logger.info(f"Task {self.id} Start at {self.simpy_env.now}, num_rev: {self.instance_sender_pipe.num_rev}.")

    def start(self):
        self.start_task_instance()
        self.run()
        self._run = True

    def restart(self, action):
        # msg = {"time": self.env.now, "type": 'stop'}
        # self.instance_sender_pipe.put(msg)
        for task in self.task_instances:
            task.process.interrupt()
        self._run = False
        if self.is_source or self.is_sink:
            self._build_ins(1)
        else:
            act = action[self.task_index]
            self._build_ins(act)
        # logger.info(f"Task {self.id} Stop at {self.simpy_env.now}.")
        yield self.simpy_env.timeout(RESTART_COST)
        self.start_task_instance()
        self._run = True

    @property
    def queued_batch(self):
        return np.array([b for t in self.task_instances for b in t.queued_batch])

    @property
    def queued_batch_size(self):
        return np.array([b.size for b in self.queued_batch])

    @property
    def queued_batch_wait_time(self):
        return np.array([(self.simpy_env.now - b.pro_time) for b in self.queued_batch])

    def batch_by_id(self, bid):
        if bid not in self.exe_batch:
            raise Exception("Queried batch id is not executing.")
        return self.exe_batch[bid]

    @property
    def utilized_ratio(self):
        return self.work_ins_num / self.parallelism

    def window_latency(self, win=5):
        if self.prev_window_latency is not None:
            if self.simpy_env.now == self.prev_window_latency[0]:
                return self.prev_window_latency[1]
        start = min(0, self.simpy_env.now - win)
        done_bth = []
        for bd in self.done_info[::-1]:
            if bd["time"] < start:
                break
            done_bth.append(bd["batch_latency"] / bd["batch_size"])
        if len(done_bth) == 0:
            self.prev_window_latency = (self.simpy_env.now, win)
        else:
            self.prev_window_latency = (self.simpy_env.now, sum(done_bth) / len(done_bth))
        return self.prev_window_latency[1]

    def batch_latency(self, batch_num=5):
        done_bth = []
        for bd in self.done_info[-batch_num:]:
            done_bth.append(bd["batch_latency"] / bd["batch_size"])
        if len(done_bth) == 0:
            return LATENCY_MAX
        else:
            return sum(done_bth) / len(done_bth)

    def upstream(self):
        parent_latencies = [p.window_latency() for p in self.parents]
        return parent_latencies

    def downstream(self):
        child_latencies = [c.window_latency() for c in self.children]
        return child_latencies

    def positional_embedding(self, dim=10):
        s_pos = self.job.topo_order[self.task_index]
        t_pos = self.job.simpy_env.now % config.obs_span

        spatial_pe = pe_helper(s_pos, dim)
        temporal_pe = pe_helper(t_pos, dim)
        return list(spatial_pe + temporal_pe)

    @property
    def np_state(self):
        state_vec = np.array(
            [*list_basic_stat(self.queued_batch_size),
             *list_basic_stat(self.queued_batch_wait_time),
             self.utilized_ratio,
             self.parallelism,
             self.window_latency(),
             *list_basic_stat(self.upstream()),
             *list_basic_stat(self.downstream()),
             self.job_utilized_ratio(),
             self.batch_latency(),
             *self.positional_embedding()
             ])
        return state_vec.reshape(1, -1)

    @property
    def id(self):
        return str(self.job.id) + '-' + str(self.task_index)

    @property
    def work_ins_num(self):
        return len([task for task in self.task_instances if not task.is_free])

    @property
    def parallelism(self):
        return len([task for task in self.task_instances if task.started])

    def job_utilized_ratio(self):
        return self.job.job_utilized_ratio

    @property
    def parents(self):
        if self._parents is None:
            if self.task_config.parent_indices is None:
                raise ValueError("Task_config's parent_indices should not be None.")
            self._parents = []
            for parent_index in self.task_config.parent_indices:
                self._parents.append(self.job.tasks_map[parent_index])
        return self._parents

    @property
    def children(self):
        if self._children is None:
            if self.task_config.children_indices is None:
                raise ValueError("Task_config's parent_indices should not be None.")
            self._children = []
            for children_index in self.task_config.children_indices:
                self._children.append(self.job.tasks_map[children_index])
        return self._children

    def ins_done_batch(self, bid):
        wrk_ins = self.wrk_ins[bid]
        # print(f"Check batch {bid}, wrk_ins: {wrk_ins}")
        for task_ins in self.task_instances:
            if task_ins.task_instance_index in wrk_ins and bid not in task_ins.completed_batch:
                return False
        return True

    def task_done_batch(self, bid):
        return bid in self.complete_batch

    def ready(self, bid):
        return all([p.task_done_batch(bid) for p in self.parents])

    def _build_ins(self, ins_num):
        assert ins_num >= 1, "Build ins num should be 1 at least."
        self.task_instances = []
        self.done_info = []
        self.instance_sender_pipe.reset()
        for task_instance_index in range(ins_num):
            self.task_instances.append(
                TaskInstance(self.simpy_env, self, task_instance_index, self.task_instance_config,
                             self.instance_sender_pipe.get_output_conn(),
                             self.instance_receiver_pipe))
        logger.info(
            f"Task {self.id} build {ins_num} ins at {self.simpy_env.now}")

    @property
    def started(self):
        for task_instance in self.task_instances:
            if not task_instance.started:
                return False
        return True


class Job:
    def __init__(self, simpy_env, job_config):
        self.simpy_env = simpy_env
        self.job_config = job_config
        self.id = job_config.job_index
        self._data_stream_pipe = None
        self._source = None
        self._sink = None
        self.tmp_pos = 0
        self._sparse_graph = None
        self._adj_m = None
        self.path_ids = None
        self._topo_order = None
        self._k_hop = None
        self._init_tasks()
        self.start_all_task()

    def _init_tasks(self):
        self.tasks_map = {}
        self.job_pipe = BroadcastPipe(self.simpy_env)
        task_configs = self.job_config.task_configs
        for task_config in task_configs.values():
            # print(task_config)
            task_index = task_config.task_index
            logger.info(task_config)
            self.tasks_map[task_index] = Task(self.simpy_env, self, task_config, self.job_pipe.get_output_conn(),
                                              self.job_pipe)
            if task_config.is_source:
                self._source = self.tasks_map[task_index]
            if task_config.is_sink:
                self._sink = self.tasks_map[task_index]
        self.source_pipe = self._source.in_pipe

    def start_all_task(self):
        for task in self.tasks:
            task.start()

    def link_from(self, data_stream):
        self._data_stream_pipe = data_stream.batch_pipe.get_output_conn()

    def run(self):
        while True:
            msg = yield self._data_stream_pipe.get()
            batch = msg["data"]
            batch.inp_time, batch.pro_time = self.simpy_env.now, self.simpy_env.now
            logger.info(f"Data Stream {batch.bid}-{batch.size} arrive at {self.simpy_env.now}")
            msg = {"time": self.simpy_env.now, "type": "down_data", "batch": batch}
            self.source_pipe.put(msg)

    def take_action(self, action):
        """
        :param action: task_ins_machine: List[task_id, task_instance_id, machine_id]
        :return:
        """
        if action.any():
            msg = {"time": self.simpy_env.now, "type": "stop", "action": action}
            self.job_pipe.put(msg)

    @property
    def tasks(self):
        return self.tasks_map.values()

    @property
    def task_num(self):
        return len(self.tasks_map)

    @property
    def used_ins_num(self):
        return np.array([task.parallelism for task in self.tasks])

    @property
    def state(self):
        return np.concatenate([self.tasks_map[i].np_state for i in range(self.task_num)])

    @property
    def adj_matrix(self):
        if self._adj_m is None:
            self._adj_m = config2adj(self.job_config)
        return self._adj_m

    def print_k_hop(self):
        job_k_nei, job_n_nei = k_hop(self.adj_matrix, 3, config.max_neis, config.max_task, config.pad_task)
        print("job_k_nei:\n", job_k_nei)
        print("job_n_nei:\n", job_n_nei)

    # @property
    # def k_hop(self):
    #     if self._k_hop is None:
    #         self._k_hop = k_hop(self.adj_matrix, K=3)
    #     return self._k_hop

    @property
    def topo_order(self):
        if self._topo_order is None:
            self._topo_order = topo_order(self.adj_matrix, self.task_num)
        return self._topo_order

    @property
    def job_utilized_ratio(self):
        work_ins = sum([task.work_ins_num for task in self.tasks])
        total_ins = sum([task.parallelism for task in self.tasks])
        return work_ins / total_ins

    @property
    def metric(self):
        if len(self._sink.sink_batch) < self.tmp_pos + 1:
            latencies = [LATENCY_MAX]
        else:
            latencies = [(batch.out_time - batch.inp_time) / batch.size
                         for batch in self._sink.sink_batch[self.tmp_pos:]]
        logger.info(f"Latency: {latencies}")
        utilized_ratio = [task.utilized_ratio for task in self.tasks]
        logger.info(f"utilized_ratio: {utilized_ratio}")
        parallelism = [task.parallelism for task in self.tasks]
        self.tmp_pos = len(self._sink.sink_batch)
        return {
            "latency": np.array(latencies),
            "utilized_ratio": utilized_ratio,
            "parallelism": np.array(parallelism)
        }

    @property
    def started(self):
        for task in self.tasks:
            if not task.started:
                return False
        return True


def config2adj(job_config):
    task_num = len(job_config.task_configs)
    adj_m = np.zeros((task_num, task_num))
    for i in range(task_num):
        for j in job_config.task_configs[i].children_indices:
            adj_m[i][j] = 1
    return adj_m


@njit
def topo_order(adj_matrix, N):
    topo_l = np.zeros(N)
    start = []
    for i in range(N):
        if adj_matrix[:, i].sum() == 0:
            start.append(i)
    topo = 0
    while len(start) > 0:
        tmp = []
        for cur in start:
            for i in range(N):
                if adj_matrix[cur, i]:
                    tmp.append(i)
                    topo_l[i] = topo + 1
        topo += 1
        start = tmp
    topo_l += 1
    return topo_l


@njit
def k_hop(adj_matrix, K, max_neis, max_task, pad_task):
    task_num = adj_matrix.shape[0]
    max_neis = max_neis
    all_K_neis = np.full((max_task, max_neis), pad_task)
    num_neis = np.zeros(max_task)
    for i in range(task_num):
        start = {i}
        k_neis = set()
        for k in range(K):
            neis = set()
            for s in start:
                neis.update(np.where(adj_matrix[s] == 1)[0])
            if len(neis) == 0:
                break
            k_neis.update(neis)
            start = neis
        k_neis = np.array(list(k_neis - {i}))
        all_K_neis[i, :len(k_neis)] = k_neis
        num_neis[i] = len(k_neis)
    return all_K_neis, num_neis


@njit
def pe_helper(position, dim=10):
    base = np.array([position / np.power(10000, 2 * (i // 2) / dim) for i in range(dim)])
    base[0::2] = np.sin(base[0::2])
    base[1::2] = np.cos(base[0::2])
    return base
