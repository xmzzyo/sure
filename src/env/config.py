import networkx as nx


class TaskInstanceConfig:
    def __init__(self, task_config):
        self.cpu = task_config.cpu
        self.duration = task_config.duration


class TaskConfig:
    def __init__(self,
                 task_index=None,
                 parallelism=1,
                 cpu=None,
                 duration=None,
                 is_source=False,
                 is_sink=False):
        self.task_index = task_index
        self.parallelism = parallelism
        self.cpu = cpu
        self.duration = duration
        self.parent_indices = set()
        self.par_conn = dict()
        self.children_indices = set()
        self.is_source = is_source
        self.is_sink = is_sink
        self.status = None

    def __str__(self):
        return f"task_index: {self.task_index}\tparallelism: {self.parallelism}\tduration: {self.duration}" \
               f"\tparent_indices: {self.parent_indices}\tchildren_indices: {self.children_indices}\t" \
               f"is_source: {self.is_source}\tis_sink: {self.is_sink}"


class JobConfig:
    def __init__(self, job_index, task_configs):
        self.job_index = job_index
        self.task_configs = task_configs
        self.edges = None

    def to_nx(self):
        G = nx.DiGraph()
        for task in self.task_configs:
            if len(task.parent_indices) == 0:
                G.add_node(task.task_index)
            for parent in task.parent_indices:
                G.add_edges_from([(parent, task.task_index)])
        return G

    @property
    def vertex_num(self):
        return len(self.task_configs)

    @property
    def edge_num(self):
        return self.edges
