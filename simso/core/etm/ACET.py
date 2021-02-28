from simso.core.etm.AbstractExecutionTimeModel \
    import AbstractExecutionTimeModel

from numpy.random import RandomState, randint


class ACET(AbstractExecutionTimeModel):
    def __init__(self, sim, nb_processors=None):
        self.sim = sim
        self.et = {}
        self.executed = {}
        self.on_execute_date = {}
        self.random = RandomState(randint(10))

    def init(self):
        pass

    def update_executed(self, job):
        if job in self.on_execute_date:
            self.executed[job] += (self.sim.now() - self.on_execute_date[job]
                                   ) * job.cpu.speed

            del self.on_execute_date[job]

    def on_activate(self, job):
        self.executed[job] = 0
        m = self.random.choice(**dict((value, proba) for value, proba in enumerate(job.p_et)))
        et = abs(self.random.normal(job.acet[m], job.et_stddev[m] / self.sim.cycles_per_ms)) * self.sim.cycles_per_ms
        self.et[job.name] = et

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
        return int(self.et[job.name] - self.get_executed(job))

    def update(self):
        for job in list(self.on_execute_date.keys()):
            self.update_executed(job)
