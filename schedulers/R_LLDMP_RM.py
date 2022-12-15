from simso.core import Scheduler
from simso.schedulers import scheduler
from simso.estimation.Diaz import conv, shrink
from itertools import combinations
import json
from read_csv import read_csv
from tqdm import trange
import difflib
from numpy import array, zeros, where, infty, random
from math import sqrt, exp, log
from scipy.stats.distributions import norm


@scheduler("simso.schedulers.R_LLDMP_RM")
class R_LLDMP_RM(Scheduler):
    """ Least Deadline Miss Probability with Rate monotonic """
    def init(self):
        self.ready_list = []
        self.activation_matrix = zeros((len(self.task_list), len(self.processors)))
        self.utilizations = array([t.utilization for t in self.task_list])
        self.deviations = array([t.deviation for t in self.task_list])
        self.max_deviations = array([t.max_deviation for t in self.task_list])

    def on_activate(self, job):
        self.ready_list.append(job)
        job.cpu.resched()

    def on_terminated(self, job):
        if job in self.ready_list:
            self.ready_list.remove(job)
        else:
            job.cpu.resched()
        self.activation_matrix[job.task.id, job.cpu.identifier] = 0

    def hoeffding(self, t, u, v):
        return exp(- t * (1 - u) ** 2 / v)

    def reward(self, job, cpu):
        if not any(self.activation_matrix[:, cpu.identifier]):
            return infty

        u = self.activation_matrix[:, cpu.identifier] @ self.utilizations # sum(self.utilizations[where(self.activation_matrix[:, cpu.identifier] > 0)[0]])
        schedulable = u + self.utilizations[job.task.id] < cpu.speed
        if not schedulable:
            return -infty
        c = 0
        u = self.activation_matrix[:job.task.id, cpu.identifier] @ self.utilizations[:job.task.id]
        u += self.utilizations[job.task.id]
        v = self.activation_matrix[:job.task.id, cpu.identifier] @ self.max_deviations[:job.task.id]
        v += self.max_deviations[job.task.id]

        try:
            c += -log(sum( p1 * self.hoeffding(p0 * cpu.speed, u, v) for p0, p1 in zip(*job.task.inter_arrival)))
        except:
            c += 0
        for k in where(self.activation_matrix[:, cpu.identifier])[0]:
            if k < job.task.id:
                continue
            #c += self.task_list[k].period * cpu.speed * (1-u)**2 / v
            try:
                c += -log(sum(p1 * self.hoeffding(p0 * cpu.speed, u, v) for p0, p1 in zip(*self.task_list[k].inter_arrival)))
            except:
                c += 0
        return c

    def allocate(self, job):
        cpu = max(self.processors, key=lambda x: self.reward(job, x))
        self.activation_matrix[job.task.id, cpu.identifier] = 1
        return cpu

    def schedule(self, cpu):
        decision = None
        if self.ready_list:
            job = min(self.ready_list, key=lambda x: x.absolute_deadline)
            if any(self.activation_matrix[job.task.id, :]) and job._was_running_on is not None:
                cpu = job._was_running_on
            else:
                cpu = self.allocate(job)
            if (cpu.running is None or
                    cpu.running.absolute_deadline > job.absolute_deadline):
                self.ready_list.remove(job)
                if cpu.running:
                    self.ready_list.append(cpu.running)
                decision = (job, cpu)
        return decision
