"""
Implementation of the Global-EDF (Earliest Deadline First) for multiprocessor
architectures.
"""
from simso.core import Scheduler
from simso.schedulers import scheduler
from simso.estimation.PreEMDF import PreEMDF
import json

@scheduler("simso.schedulers.EMDF")
class EMDF(Scheduler):
    """Earliest Modal Deadline First"""
    def init(self):
        with open('./EMDF.json') as f:
            dict_params = json.load(f)
        self.clf = PreEMDF(n_tasks=len(self.task_list))
        self.clf.set_parameters(dict_params)

    def on_activate(self, job):
        job.cpu.resched()

    def on_terminated(self, job):
        job.cpu.resched()

    def schedule(self, cpu):
        # List of ready jobs not currently running:
        ready_jobs = [t.job for t in self.task_list
                      if t.is_active() and not t.job.is_running()]
        previous_cluster = self.clf.predict(self.ARTT)
        quantiles = [0] * len(self.task_list)
        for i, r in enumerate(self.ARTT):
            quantiles[i] = self.clf.mixture_models[previous_cluster][i].predict_quantile(r)

        if ready_jobs:
            # Select a free processor or, if none,
            # the one with the greatest deadline (self in case of equality):
            key = lambda x: (
                1 if not x.running else 0,
                x.running.modal_deadline if x.running else 0,
                1 if x is cpu else 0
            )
            cpu_min = max(self.processors, key=key)

            # Select the job with the least priority: # MARC ANTOINE
            job = min(ready_jobs, key=lambda x: x.modal_deadline)

            if (cpu_min.running is None or
                    cpu_min.running.modal_deadline > job.modal_deadline):
                print(self.sim.now(), job.name, cpu_min.name)
                return (job, cpu_min)
