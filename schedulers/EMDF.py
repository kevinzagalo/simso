"""
Implementation of the Global-EDF (Earliest Deadline First) for multiprocessor
architectures.
"""
from simso.core import Scheduler
from simso.schedulers import scheduler
from simso.estimation.KmeansInertia import KmeansInertia
from simso.estimation import rInvGaussMixture
import numpy as np
import json
import os
from scipy.stats import chi2

class PreEMDF:

    def __init__(self, inertia=0.1, n_tasks=None, n_components_max=5, alpha=None):
        self.inertia = inertia
        self.alpha = alpha
        self.igmms = {}
        self.kmeans_inertia = None
        self.n_components_max = n_components_max
        self.n_tasks_ = n_tasks
        self.mixture_models = {}
        self.dmp = {}
        self.transitions = None

    def fit(self, schedule):
        self.n_tasks_ = len(schedule.task_list)
        activation_list = np.array([x[0] for x in schedule.response_times if all(x[1])])
        response_times = np.array([x[1] for x in schedule.response_times if all(x[1])])

        ## Clustering
        self.kmeans_inertia = KmeansInertia(inertia=self.inertia).fit(response_times)
        clusters = self.kmeans_inertia.predict(response_times)

        ## Markov transitions
        k = len(self.kmeans_inertia.model.cluster_centers_)
        self.transitions = np.zeros((k, k))
        for (i, j) in zip(clusters[:-1], clusters[1:]):
            self.transitions[i, j] += 1
        for i in range(k):
            self.transitions[i, :] /= self.transitions[i, :].sum()

        ## Parametric estimation
        for k in set(clusters):
            self.mixture_models[k] = {}
            self.dmp[k] = {}
            for task in schedule.task_list:
                bic_list = []
                igmm_list = []
                ind_k_task = [p1 and p2 for p1, p2 in zip(list(clusters == k), list(activation_list == task.id))]
                for n_components in range(1, self.n_components_max):
                    igmm_list.append(rInvGaussMixture(n_components=n_components).fit(response_times[ind_k_task, task.id]))
                    bic_list.append(igmm_list[-1].bic(response_times[ind_k_task, task.id]))
                best = np.argmax(bic_list)
                self.mixture_models[k][task.id] = igmm_list[best]
                self.dmp[k][task.id] = [0] * best
                for component in range(best):
                    print(k, task.id, component, self.deadline_miss_proba(k, task, component))
                    self.dmp[k][task.id][component] = self.deadline_miss_proba(k, task, component)
        return self

    def deadline_miss_proba(self, k, task, component):
        #q = self.mixture_models[k][task.id].quantile(task.alpha, component)
        return self.mixture_models[k][task.id].dmp(task.deadline, component)

    def _predict(self, x, task_id): # X=[1,2,3,4]
        cluster = self.kmeans_inertia.predict([x])[0] #([x])
        task_component = self.mixture_models[cluster][task_id].predict([x[task_id]])[0]
        return cluster, task_component

    def predict(self, X, task_id): # X=[[1,2,3,4],...,[]]
        if len(np.shape(X)) == 1:
            return self._predict(X, task_id)
        elif np.shape(X)[0] == 1:
            return self._predict(X, task_id)
        else:
            return [self._predict(x, task_id) for x in X]

    def get_parameters(self):
        params = {}
        for k, centroids in enumerate(self.kmeans_inertia.model.cluster_centers_):
            params[k] = {}
            params[k]["centroids"] = list(centroids)
            for n in range(self.n_tasks_):
                params[k][n] = {}
                params[k][n]["igm_param"] = self.mixture_models[k][n].get_parameters()
                params[k][n]["dmp"] = self.dmp[k][n]
        # params["kmeans_params"] = self.kmeans_inertia.get_parameters()
        params["n_tasks_"] = self.n_tasks_
        params["inertia"] = self.inertia
        return params

    def set_parameters(self, params):
        self.n_tasks_ = params["n_tasks_"]
        self.n_clusters_ = len(params.keys())
        self.kmeans_inertia = KmeansInertia(inertia=self.inertia)
        # self.kmeans_inertia.set_parameters(params["kmeans_params"])
        self.mixture_models = {}
        for k in range(self.n_clusters_):
            self.mixture_models[k] = {}
            self.mixture_models[k]["centroids"] = params[k]["centroids"]
            for n in range(self.n_tasks_):
                self.mixture_models[k][n]["igm_param"] = rInvGaussMixture(n_components=params[k][n]["n_components"])
                self.mixture_models[k][n]["igm_param"] .set_parameters(params[k][n]["igm_param"])
                self.dmp[k][n] = params[k][n]["dmp"]

    def save(self, path):
        with open(os.path.join(path, 'EMDF.json'), 'w') as f:
            json.dump(self.get_parameters(), f, indent=6)

    def load(self, path_json_file):
        with open(path_json_file) as f:
            self.set_parameters(json.load(f))
        return self


@scheduler("simso.schedulers.EMDF")
class EMDF(Scheduler):
    """Earliest Modal Deadline First"""
    def init(self):
        try:
            with open('./EMDF.json') as f:
                dict_params = json.load(f)
        except:
            print('Train the model first !')
        self.clf = PreEMDF()
        self.clf.set_parameters(dict_params)

    def on_activate(self, job):
        job.cpu.resched()

    def on_terminated(self, job):
        job.cpu.resched()

    def schedule(self, cpu):
        # List of ready jobs not currently running:
        ready_jobs = [t.job for t in self.task_list
                      if t.is_active() and not t.job.is_running()]

        if ready_jobs:
            # Select a free processor or, if none,
            # the one with the greatest deadline (self in case of equality):
            for j in ready_jobs:
                if all(self.ARTT):
                    previous_cluster, component = self.clf.predict(self.ARTT, j.task.id)
                    j.dmp = self.dmp[previous_cluster][j.task.id][component]
                else:
                    j.dmp = j.absolute_deadline

            key = lambda x: (
                1 if not x.running else 0,
                x.running.dmp if x.running else 0,
                1 if x is cpu else 0
            )
            cpu_min = max(self.processors, key=key)

            # Select the job with the least priority: # MARC ANTOINE
            job = min(ready_jobs, key=lambda x: x.dmp)

            if (cpu_min.running is None or
                    cpu_min.running.dmp > job.dmp):
                print(self.sim.now(), job.name, cpu_min.name)
                return (job, cpu_min)
