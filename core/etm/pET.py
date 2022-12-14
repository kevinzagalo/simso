from simso.core.etm.AbstractExecutionTimeModel  import AbstractExecutionTimeModel
from numpy import random


class pET(AbstractExecutionTimeModel):
    def __init__(self, sim, nb_processors=None):
        self.sim = sim
        self.et = {}
        self.executed = {}
        self.on_execute_date = {}
        #self.random = RandomState(randint(10))

    def init(self):
        pass

    def update_executed(self, job):
        if job in self.on_execute_date:
            self.executed[job] += (self.sim.now() - self.on_execute_date[job]
                                   ) * job.cpu.speed

            del self.on_execute_date[job]

    def on_activate(self, job):
        self.executed[job] = 0
        if not isinstance(job.task.modes, list):
            modes = [job.task.modes]
        else:
            modes = job.task.modes
        if not isinstance(job.task.proba, list):
            proba = [job.task.proba]
        else:
            proba = job.task.proba
        w = random.choice(a=modes, p=proba)
        self.et[job] = w

    def on_execute(self, job):
        self.on_execute_date[job] = self.sim.now()

    def on_preempted(self, job):
        self.update_executed(job)

    def on_terminated(self, job):
        self.update_executed(job)

    def on_abort(self, job):
        self.update_executed(job)

    def get_executed(self, job):
        if job in self.on_execute_date:
            c = (self.sim.now() - self.on_execute_date[job]) * job.cpu.speed
        else:
            c = 0
        return self.executed[job] + c

    def get_ret(self, job):
        return int(self.et[job] - self.get_executed(job))

    def update(self):
        for job in list(self.on_execute_date.keys()):
            self.update_executed(job)