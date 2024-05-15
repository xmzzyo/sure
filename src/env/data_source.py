import calendar
import gzip
import os
import pickle
import re

import numpy as np
import pandas as pd

from src.env.job import BroadcastPipe, Batch
from src.utils.config_parser import config
from src.utils.logger import logger
from src.utils.utils import DATA_DIR

GEN_INTERVAL = config.data_src_itv


class RequestGenerator:
    def __init__(self):
        self.data_source = None
        self.idx, self.cur_idx = 0, 0

    def load_data(self):
        pass

    def gen(self):
        pass


class ClarkGenerator(RequestGenerator):
    def __init__(self):
        super().__init__()
        self.data_source = self.load_data()

    def load_data(self):
        requests_stream = []
        # "../../data/"  DATA_DIR
        with open(os.path.join(DATA_DIR, "clarknet-min.txt"), "r") as f:
            for req in f.readlines():
                requests_stream.append(int(req))
        requests_stream = np.array(requests_stream)
        m, s = divmod(len(requests_stream), 60)
        h, m = divmod(m, 60)
        logger.info(f"Request Num: {len(requests_stream)}, Duration: %02dH:%02dm:%02ds" % (h, m, s))
        return requests_stream

        # "../../data/"  DATA_DIR
        pkl_path = os.path.join(DATA_DIR, "clarknet-min.dat")
        if not os.path.exists(pkl_path):
            logger.info("pkl data not found, create requests data from scratch...")
            file_path = ["data/clarknet_access_log_Aug28.gz", "data/clarknet_access_log_Sep4.gz"]
            data = []
            for fp in file_path:
                with gzip.open(fp, 'r') as pf:
                    for line in pf:
                        try:
                            dat = \
                                re.search(r"\d{2}/[a-zA-Z]+?/\d{4}(\-|\/|.)\d{2}\1\d{2}\1\d{2}",
                                          line.decode(encoding="utf-8"))[0]
                        except:
                            print(fp, line)
                        m = dat.split("/")[1]
                        if m in calendar.month_abbr:
                            md = list(calendar.month_abbr).index(m)
                        elif re.match(r"\d+", m):
                            md = m
                        else:
                            continue
                        dat = re.sub(m, str(md), dat)
                        data.append(dat)
            data = pd.DataFrame(data, columns=["date"])
            data["date"] = pd.to_datetime(data["date"], format="%d/%m/%Y:%H:%M:%S")
            requests_stream = data.resample('1T', on='date').count().values.squeeze()
            pickle.dump(requests_stream, open(pkl_path, "wb"))
        requests_stream = []
        with open(os.path.join(DATA_DIR, "clarknet-min.txt"), "r") as f:
            for req in f.readlines():
                requests_stream.append(int(req))
        requests_stream = np.array(requests_stream)
        m, s = divmod(len(requests_stream), 60)
        h, m = divmod(m, 60)
        logger.info(f"Request Num: {len(requests_stream)}, Duration: %02dH:%02dm:%02ds" % (h, m, s))
        return requests_stream

    def gen(self):
        self.cur_idx = 0  # random.randint(0, len(self.data_source) - 1)
        while True:
            yield self.data_source[self.cur_idx]
            self.cur_idx += 1
            if self.cur_idx == len(self.data_source):
                self.cur_idx = 0

    def plot(self):
        import matplotlib.pyplot as plt
        font = {'family': 'Times New Roman',
                'weight': 'bold',
                'size': 35,
                }
        plt.figure(figsize=(12, 7))
        plt.plot(self.data_source[:2000])
        plt.xlabel("Time (minutes)", font)
        plt.ylabel("#Requests", font)
        plt.yticks(fontproperties='Times New Roman', size=30)
        # plt.xticks([0, 5000, 10000, 15000, 20000], fontproperties='Times New Roman', size=30)
        plt.savefig("../../files/workload.pdf", bbox_inches='tight')
        plt.show()


class PoissonGenerator(RequestGenerator):
    def gen(self):
        while True:
            yield np.random.poisson(3) + 1


class DataStream:
    def __init__(self, data_gen=None):
        self._env = None
        self.simulation = None
        self.cluster = None
        self.destroyed = False
        self.data_gen = ClarkGenerator() if data_gen is None else data_gen
        self.batch_pipe = None
        self.data_log = []

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def reset(self, env):
        self._env = env
        self.batch_pipe = BroadcastPipe(env)
        self.data_log = []

    def run(self):
        batch_idx = 0
        for batch in self.data_gen.gen():
            if batch == 0:
                continue
            self.data_log.append(batch)
            data = Batch(bid=batch_idx, inp_time=self._env.now, size=batch)
            batch_idx += 1
            msg = {"time": self._env.now, "type": "data_src", "data": data}
            self.batch_pipe.put(msg)
            # print(f"Data Stream {batch} arrive at {self._env.now}")
            yield self._env.timeout(GEN_INTERVAL)
        self.destroyed = True

    def schedule(self):
        self._env.process(self.run())


if __name__ == "__main__":
    clark = ClarkGenerator()
    print(np.mean(clark.data_source))
    clark.plot()
    exit()
    env = simpy.Environment()
    ds = DataStream()
    ds.reset(env)
    env.process(ds.run())
    env.run(until=25)
