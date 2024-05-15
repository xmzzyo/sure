import os
import pickle
import re

import pandas as pd
from tqdm import tqdm

from src.env.config import JobConfig, TaskConfig
from src.utils.config_parser import config

MIN_TASK_NUM = 0


class CSVReader:
    def __init__(self, filename, trace="ali_v18"):
        """
        count    90.000000
        mean     20.466667
        std       9.739529
        min      12.000000
        25%      13.250000
        50%      15.000000
        75%      25.000000
        max      46.000000
        dtype: float64
        """
        self.filename = filename
        print("Reading jobs from ", filename)
        # self._ali_v18()
        self.job_configs = self.sample()
        post_job_configs = []
        for i in range(len(self.job_configs)):
            jc = self.job_configs[i]
            task_num = len(jc.task_configs) - 2
            if task_num < MIN_TASK_NUM:
                continue
            p = max(int(config.max_task_ins / 2), 1)
            for t in range(1, task_num + 1):
                jc.task_configs[t].parallelism = p
                duration = jc.task_configs[t].duration
                jc.task_configs[t].duration = max(0, duration) / config.duration_norm
            post_job_configs.append(jc)
        self.job_configs = sorted(post_job_configs, key=lambda x: len(x.task_configs))

    def _ali_v17(self):
        pass

    def _ali_v18(self):
        self.job_configs = pickle.load(open(self.filename, "rb"))
        return
        df = pd.read_csv(self.filename,
                         names=["task_name", "instance_num", "job_name", "task_type", "status", "start_time",
                                "end_time", "plan_cpu", "plan_mem"])

        # df.instance_num = df.instance_num.astype(dtype=int)
        job_task_map = {}
        for i in tqdm(range(len(df)), total=len(df)):
            series = df.iloc[i]
            # print(series.job_name)
            if not re.match("[A-Z]\\d+(_\\d+)*", series.task_name):
                continue
            job_id = series.job_name
            try:
                task_ids = series.task_name.split("_")
                task_id = int(re.sub("[A-Z]", "", task_ids[0]))
                parent_indices = [int(ti) for ti in task_ids[1:] if re.match("\\d+", ti)]
                cpu = series.plan_cpu
                status = series.status
                start_time = series.start_time
                end_time = series.end_time
                instances_num = int(series.instance_num)
            except Exception as e:
                continue

            task_configs = job_task_map.setdefault(job_id, dict())
            task_configs[task_id] = TaskConfig(task_index=task_id, parallelism=instances_num, cpu=cpu,
                                               duration=end_time - start_time)
            task_configs[task_id].parent_indices = set(parent_indices)
            task_configs[task_id].status = status

        job_configs = []
        for job_id, task_configs in job_task_map.items():
            task_configs = self.preprocess_job(task_configs)
            if task_configs == -1:
                continue
            job_configs.append(JobConfig(job_id, task_configs))

        self.job_configs = job_configs
        print(len(job_configs))
        pickle.dump(job_configs, open("data/ali_v18.dat", "wb"))

    @staticmethod
    def preprocess_job(task_configs):
        # Ready | Waiting | Running | Terminated | Failed | Cancelled
        for task in task_configs.values():
            status = task.status
            if status in ["Ready", "Waiting", "Failed", "Cancelled"]:
                print("Failed job.")
                return -1
        if any(["Terminated" not in task.status for task in task_configs.values()]):
            return -1
        if list(sorted(task_configs.keys())) != list(range(1, len(task_configs) + 1)):
            print("task index exception.")
            return -1
        try:
            for task in task_configs.values():
                for parent_idx in task.parent_indices:
                    task_configs[parent_idx].children_indices.add(task.task_index)
        except KeyError:
            print("task index KeyError.")
            return -1

        source_ids = set()
        sink_ids = set()
        for task in task_configs.values():
            if len(task.parent_indices) == 0:
                source_ids.add(task.task_index)
            if len(task.children_indices) == 0:
                sink_ids.add(task.task_index)
        intersect = source_ids.intersection(sink_ids)
        if len(source_ids) == 0 or len(sink_ids) == 0 or len(intersect) > 0:
            print(len(source_ids), len(sink_ids), len(intersect), len(task_configs))
            return -1

        virtual_src = TaskConfig(task_index=0, cpu=0, duration=0, is_source=True)
        virtual_src.children_indices = source_ids
        for task_id in source_ids:
            task_configs[task_id].parent_indices.add(0)

        virtual_sink = TaskConfig(task_index=len(task_configs) + 1, cpu=0, duration=0, is_sink=True)
        virtual_sink.parent_indices = sink_ids
        for task_id in sink_ids:
            task_configs[task_id].children_indices.add(len(task_configs) + 1)

        task_configs[virtual_src.task_index] = virtual_src
        task_configs[virtual_sink.task_index] = virtual_sink

        return task_configs

    def sample(self):
        min_duration = 2000
        # f = re.sub("ali_v18", f"ali_v18-D{min_duration}", self.filename)
        f = self.filename
        if os.path.exists(f):
            return pickle.load(open(f, "rb"))
        sampled = []
        for job in self.job_configs:
            d = sorted(list(set([max(t.duration, 1) for t in job.task_configs.values()])))
            if len(d) == 1:
                d = d[0]
            else:
                d = d[1]
            if d < min_duration:
                continue
            sampled.append(job)
        # sampled = np.random.choice(self.job_configs, num)
        pickle.dump(sampled, open(f, "wb"))
        return sampled
