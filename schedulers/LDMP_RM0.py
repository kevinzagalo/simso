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

class PreGRM:

    def __init__(self, execution_times, path):#, periods):
        C = [(i, c) for i, c in enumerate(execution_times)]
        dict_mu = {i: c for i, c in C}
        for n in trange(2, len(execution_times)):
            for CC in combinations(C, n):
                key = ""
                #u = 0
                mu = ([0], [1.])

                for (i, c) in CC:
                    key += str(i)
                #    u += sum([cc[0] * cc[1] for cc in zip(*c)]) / periods[i]
                #if u > 1:
                #    continue

                #for ks in dict_mu.keys():
                #    ks_str = str(ks)
                #    task_combinations = str(key)
                #    if ks_str in task_combinations:
                #        mu = dict_mu[ks]
                #        for s in difflib.ndiff(ks_str, task_combinations):
                #            if s[-1] in ('[', ",", " ", ']'):
                #                continue
                #            if s[0] == '+':
                #                mu = conv(execution_times[int(s[-1])], mu)
                #    else:
                for (i, c) in CC:
                    mu = conv(c, mu)

                dict_mu[key] = mu
        with open(path, 'w') as json_file:
            json.dump(dict_mu, json_file, indent=1)


@scheduler("simso.schedulers.LDMP_RM0")
class LDMP_RM0(Scheduler):
    """ Least Deadline Miss Probability with Rate monotonic """
    def init(self):
        self.ready_list = []
        self.activation_matrix = zeros((len(self.task_list), len(self.processors)))
        self.utilizations = array([t.utilization for t in self.task_list])
        self.deviations = array([t.deviation for t in self.task_list])
        self.max_deviations = array([t.max_deviation for t in self.task_list])
        with open("/home/kzagalo/Documents/rInverseGaussian/simso/schedulers/GRM.json") as json_file:
            self.dict_mu = json.load(json_file)

    def on_activate(self, job):
        self.ready_list.append(job)
        job.cpu.resched()

    def on_terminated(self, job):
        if job in self.ready_list:
            self.ready_list.remove(job)
        else:
            job.cpu.resched()
        self.activation_matrix[job.task.id, job.cpu.identifier] = 0

    def ids_to_str(self, ids):
        return str(ids)[1:-1].replace(',', '').replace(' ', '')

    def dmp(self, b, u, v, t, eps=1e-200):
        return max((eps, norm.cdf(-((1-u)*t - b)/sqrt(v*t)) - exp(-2*b*(1-u)/v) * norm.cdf(-((1-u)*t + b)/sqrt(v*t))))

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
        processor_ids = where(self.activation_matrix[:job.task.id, cpu.identifier])[0]
        delta = 0
##
        if any(self.activation_matrix[:job.task.id, cpu.identifier]):
            higher_priority_tasks = array(self.task_list)[processor_ids]
            for t in higher_priority_tasks:
                delta += t.job.computation_time_cycles

        key = self.ids_to_str(processor_ids) + str(job.task.id)
        if key == '':
            pass
        else:
            mu = shrink(self.dict_mu[key],  delta + job.computation_time_cycles)
            c = -log(sum([sum([(self.dmp(b0/cpu.speed, u, v, p0)) * b1 for b0, b1 in zip(*mu)]) * p1
                         for p0, p1 in zip(*job.task.inter_arrival)]))
        for k in where(self.activation_matrix[:, cpu.identifier])[0]:
            if k <= job.task.id:
                continue
            processor_ids_k = where(self.activation_matrix[:k+1, cpu.identifier])[0]
            higher_priority_tasks = array(self.task_list)[processor_ids_k]
            key = self.ids_to_str(processor_ids_k)
            if key == '':
                continue
            delta = 0
            for t in higher_priority_tasks:
                delta += t.job.computation_time_cycles
            mu = conv(shrink(self.dict_mu[key],  delta), (job.task.modes, job.task.proba))
            u = (self.activation_matrix[:k+1, cpu.identifier] @ self.utilizations[:k+1] + self.utilizations[job.task.id])
            v = (self.activation_matrix[:k+1, cpu.identifier] @ self.max_deviations[:k+1] + self.max_deviations[job.task.id])
            c += -log(sum([sum([(self.dmp(b0/cpu.speed, u, v, p0)) * b1 for b0, b1 in zip(*mu)]) * p1
                         for p0, p1 in zip(*self.task_list[k].inter_arrival)]))
        return c

    def schedule(self, cpu):
        decision = None
        if self.ready_list:
            job = min(self.ready_list, key=lambda x: x.period)
            #if any(self.activation_matrix[job.task.id, :]) and job._was_running_on is not None:
            #    cpu = job._was_running_on
            #else:
            cpu = max(self.processors, key=lambda x: self.reward(job, x))
            for proc in self.processors:
                self.activation_matrix[job.task.id, proc.identifier] = int(proc.identifier == cpu.identifier)
            if (cpu.running is None or
                    cpu.running.absolute_deadline > job.absolute_deadline):
                self.ready_list.remove(job)
                if cpu.running:
                    self.ready_list.append(cpu.running)
                decision = (job, cpu)
        return decision
