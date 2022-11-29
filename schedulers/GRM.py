from simso.core import Scheduler
from simso.schedulers import scheduler
from simso.estimation.Diaz import conv, shrink
from itertools import combinations
import json
from read_csv import read_csv
from tqdm import trange
import difflib
from numpy import array, zeros, where, infty
from math import sqrt, exp, log
from scipy.stats.distributions import norm

class PreGRM:

    def __init__(self, execution_times, periods):
        C = [(i, c) for i, c in enumerate(execution_times)]
        dict_mu = {i: c for i, c in C}
        for n in trange(2, len(execution_times)):
            for CC in combinations(C, n):
                key = ""
                u = 0
                mu = ([0], [1.])

                for (i, c) in CC:
                    key += str(i)
                    u += sum([cc[0] * cc[1] for cc in zip(*c)]) / periods[i]
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
        with open("/home/kzagalo/Documents/rInverseGaussian/simso/schedulers/GRM.json", 'w') as json_file:
            json.dump(dict_mu, json_file, indent=1)


@scheduler("simso.schedulers.GRM")
class GRM(Scheduler):
    """ Rate monotonic """
    def init(self):
        self.ready_list = []
        self.activation_matrix = zeros((len(self.task_list), len(self.processors)))
        self.utilizations = array([t.utilization for t in self.task_list])
        self.deviations = array([t.deviation for t in self.task_list])
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

    def dmp(self, b, u, v, t, eps=1e-100):
        return max((eps, norm.cdf(-((1-u)*t - b)/sqrt(v*t)) - exp(-2*b*(1-u)/v) * norm.cdf(-((1-u)*t + b)/sqrt(v*t))))

    def reward(self, job, cpu):
        processor_ids = where(self.activation_matrix[:, cpu.identifier] > 0)[0]
        if len(processor_ids) == 0:
            return infty

        u = self.activation_matrix[:, cpu.identifier] @ self.utilizations / cpu.speed# sum(self.utilizations[where(self.activation_matrix[:, cpu.identifier] > 0)[0]])
        if u > 1 or u + self.utilizations[job.task.id] > 1:
            return 0
        processor_ids = where(self.activation_matrix[:job.task.id, cpu.identifier] > 0)[0]
        key = str(processor_ids)[1:-1].replace(',', '')
        key = key.replace(' ', '')
        if key == '':
            c = 0
        else:
            mu = self.dict_mu[key]
            u = self.activation_matrix[:job.task.id, cpu.identifier] @ self.utilizations[:job.task.id] / cpu.speed
            v = self.activation_matrix[:job.task.id, cpu.identifier] @ self.deviations[:job.task.id] / cpu.speed**2
            c = -log(sum([self.dmp(b0/cpu.speed, u, v, job.task.period) * b1 for b0, b1 in zip(*mu)]))
        for k in processor_ids:
            processor_ids_k = where(self.activation_matrix[:k, cpu.identifier] > 0)[0]
            key = str(processor_ids_k)[1:-1].replace(',', '')
            key = key.replace(' ', '')
            if key == '':
                continue
            mu = self.dict_mu[key]
            delta = 0
            for l in processor_ids_k:
                delta += self.task_list[l].job.actual_computation_time
            mu = shrink(mu, cpu.speed *


                        delta)
            mu = conv(mu, (job.task.modes, job.task.proba))
            u = (self.activation_matrix[:k, cpu.identifier] @ self.utilizations[:k] + self.utilizations[job.task.id])/cpu.speed
            v = (self.activation_matrix[:k, cpu.identifier] @ self.deviations[:k] + self.deviations[job.task.id])/cpu.speed**2
            c += -log(sum([self.dmp(c0/cpu.speed, u, v, self.task_list[k].period) * c1 for c0, c1 in zip(*mu)]))
        return c

    def schedule(self, cpu):
        decision = None
        if self.ready_list:
            # Job with highest priority.
            job = min(self.ready_list, key=lambda x: x.period)

            # Get a free processor or a processor running a low priority job.
            if any(self.activation_matrix[job.task.id, :]) and job._was_running_on is not None:
                cpu = job._was_running_on
            else:
                cpu = max(self.processors, key=lambda x: self.reward(job, x))
                self.activation_matrix[job.task.id, cpu.identifier] = 1

            if (cpu.running is None or
                    cpu.running.period > job.period):
                self.ready_list.remove(job)
                if cpu.running:
                    self.ready_list.append(cpu.running)
                decision = (job, cpu)

        return decision


if __name__ == "__main__":

    #c1 = ([3, 6], [.5, .5])
    #c2 = ([2, 3], [.6, .4])
    #c3 = ([3, 6], [.7, .3])
    #c4 = ([5, 10], [.9, .1])
    #c5 = ([6, 12], [.2, .8])
    #c6 = ([3, 9], [.25, .75])
    #execution_times = (c1, c2, c3, c4, c5, c6)
    #periods = list(range(10, 20))
    c1 = ([3, 6], [.5, .5])
    c2 = ([10, 12], [.6, .4])
    c3 = ([8, 13], [.7, .3])
    c4 = ([10, 20], [.9, .1])
    c5 = ([40, 52], [.2, .8])
    c6 = ([30, 58], [.25, .75])
    execution_times = (c1, c2, c3, c4, c5, c6)
    periods = list(range(100, 2000))
    u = sum([sum([cc0*cc1 for cc0, cc1 in zip(*c)]) / periods[i] for i, c in enumerate(execution_times)])
    #x = sorted(list(zip(execution_times, periods)), key=lambda x: x[1])
    #execution_times, periods = array(x).T
    PreGRM(execution_times, periods)
