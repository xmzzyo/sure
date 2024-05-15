import json
import time

from src.utils.config_parser import config

MOI_ITV = config.mont_itv


class Monitor(object):
    def __init__(self, env):
        self.env = env
        self.job = None
        self.event_file = None
        self.events = []

    def reset(self, job=None):
        if job is not None:
            self.job = job
        self.event_file = f"{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}.event"
        self.events = []

    def run(self):
        while True:
            yield self.env.timeout(MOI_ITV)
            if not self.job.started:
                continue
            state = {
                "timestamp": self.env.now,
                "job_state": self.job.state,
                "metrics": self.job.metric
            }
            self.events.append(state)
        # self.write_to_file()

    def write_to_file(self):
        with open(self.event_file, 'w') as f:
            json.dump(self.events, f, indent=4)
