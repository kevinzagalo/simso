"""
Implementation of EDF-MD.
"""
from simso.core import Scheduler
from simso.schedulers import scheduler


@scheduler("simso.schedulers.EDF_CD")
class EDF_MD(Scheduler):

    def init(self):
        self.ready_list = []

    def on_activate(self, job):
        if all(self.ARTT):
            job.priority = self.modes.conditional_dmp(self.ARTT, job.task.id, self.last_activated_jobs)
        else:
            job.priority = job.absolute_deadline
        self.ready_list.append(job)
        job.cpu.resched()

    def on_terminated(self, job):
        if job in self.ready_list:
            self.ready_list.remove(job)
        else:
            job.cpu.resched()

    def schedule(self, cpu):

        if self.ready_list:
            # Key explanations:
            # First the free processors
            # Among the others, get the one with the greatest deadline
            # If equal, take the one used to schedule

            key = lambda x: (0 if not x.running else 1,
                             self.modes.conditional_dmp(self.ARTT, x.running.task.id,
                                                        self.last_activated_jobs) if x.running and all(self.ARTT) else 1,
                             x.running.absolute_deadline if x.running else 1,
                             0 if x is cpu else 1)

            cpu_min = min(self.processors, key=key)

            active_jobs = [j for j in self.ready_list if j.is_active()]

            if active_jobs:
                key = lambda x: (self.modes.conditional_dmp(self.ARTT, x.task.id, self.last_activated_jobs)
                                 if all(self.ARTT) else 0,
                                 x.priority)
                job = min(active_jobs, key=key)


                if (cpu_min.running is None or
                        self.modes.conditional_dmp(self.ARTT, cpu_min.running.task.id,
                                                   self.last_activated_jobs) > self.modes.conditional_dmp(self.ARTT, job.task.id,
                                                                                                          self.last_activated_jobs)):
                    self.ready_list.remove(job)
                    if cpu_min.running:
                        self.ready_list.append(cpu_min.running)
                    return job, cpu_min
