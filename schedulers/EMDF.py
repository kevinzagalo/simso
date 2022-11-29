"""
Implementation of the Global-EDF (Earliest Deadline First) for multiprocessor
architectures.
"""
from simso.core import Scheduler
from simso.schedulers import scheduler
from simso.estimation.KmeansInertia import KmeansInertia
from simso.estimation import RTInvGaussMixture as rInvGaussMixture
import numpy as np
import json
import os
from scipy.stats import chi2
import matplotlib.pyplot as plt


class PreEMDF:

    def __init__(self, verbose=False, inertia=0.1, n_tasks=None, n_components_max=5, alpha=None):
        self.inertia = inertia
        self.alpha = alpha
        self.igmms = {}
        self.kmeans_inertia = None
        self.n_components_max = n_components_max
        self.n_tasks_ = n_tasks
        self.mixture_models = {}
        self.dmp = {}
        self.transitions = None
        self.verbose = verbose

    def fit(self, schedule):
        self.n_tasks_ = len(schedule.task_list)
        X = np.array(schedule.response_times, dtype=object).copy()
        utilizations, deviations = [], []
        for task in schedule.task_list:
            m = sum([c0 * c1 for c0, c1 in zip(task.modes, task.proba)])
            utilizations.append(m / task.period)
            deviations.append(sum([(c0 - m) ** 2 * c1 for c0, c1 in zip(task.modes, task.proba)]) / task.period)
        deviations = np.cumsum(deviations)
        utilizations = np.cumsum(utilizations)
        cvs = deviations / (1 - utilizations) ** 2

        for task in schedule.task_list[1:]:
            bic_list = []
            igmm_list = []
            ind_k_task = np.where(X[:, 0] == task.id)[0]
            r_task = [r[task.id] for r in X[ind_k_task, 1]]
            for n_components in range(1, 3):# self.n_components_max):
                if self.verbose:
                    print('Task', task.id, 'component', n_components)
                rIG = rInvGaussMixture(n_components=n_components, cv_init=cvs[task.id-1], verbose=self.verbose,
                                       utilization=utilizations[task.id-1]).fit(r_task)
                igmm_list.append(rIG)
                bic_list.append(rIG.bic(r_task))
            best = np.argmax(bic_list)
            self.mixture_models[task.id] = igmm_list[best]
            self.dmp[task.id] = [0] * (best+1)
            for component in range(best+1):
                self.dmp[task.id][component] = self.mixture_models[task.id].dmp(task.deadline, component)

        backlogs = np.zeros((len(X), self.n_tasks_))
        for n_transition, job in enumerate(schedule.response_times):
            task, response_times, arrival = job
            for i, r in enumerate(response_times):
                if i == 0:
                    continue
                backlogs[n_transition, i] = self.rt_to_bl(task, i, r)

        for i in range(self.n_tasks_):
            plt.hist(backlogs[:, i], bins=25)
            plt.show()

        return self

    def reward(self, b, b_prime, task_id, task_list):
        r = sum([self.mixture_models[i].dmp(task_list[i].deadline, b[i]) for i in range(1, task_id)])
        r += sum([self.mixture_models[i].dmp(task_list[i].deadline, b_prime[i]) for i in range(task_id+1, self.n_tasks_)])
        return np.exp(-r)

    def rt_to_bl(self, new_task, task, rt):
        component = self.mixture_models[task].predict(rt)
        b = self.mixture_models[task].backlog_[component]
        if task > new_task:
            return b
        else:
            return 0

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
        self.activation_matrix = np.zeros((len(self.task_list), len(self.processors)))
        self.utilizations = np.array([t.utilization for t in self.task_list])
        self.backlogs = np.zeros((len(self.task_list), len(self.processors)))

    def on_activate(self, job):
        job.cpu.resched()

    def on_terminated(self, job):
        job.cpu.resched()
        self.activation_matrix[job.task.id, job.cpu._identifier] = 0

    def schedule(self, cpu):
        decision = None
        if self.ready_list:
            # Get a free processor or a processor running a low priority job.
            key = lambda x: (
                0 if x.running is None else 1,
                0 if self.activation_matrix[x._identifier] @ self.utilizations + x.running.task.utilization > 1 else 1,
                -x.running.period if x.running else 0,
                0 if x is cpu else 1
            )

            components = [self.clf.mixture_models[i].predict(r) for i, r in enumerate(self.response_times[-1][1])]
            b = np.array([self.clf.mixture_models[i].backlogs_[c] for i, c in enumerate(components)])

            #for j in self.ready_list:
            #    component = self.clf.predict(j.task.response_times[—1])
            #    self.backlogs[j.]
            cpu_min = min(self.processors, key=key)

            # Job with highest priority.
            job = min(self.ready_list, key=lambda x: x.period)

            if (cpu_min.running is None or
                    cpu_min.running.period > job.period):
                self.ready_list.remove(job)
                if cpu_min.running:
                    self.ready_list.append(cpu_min.running)
                decision = (job, cpu_min)
                self.activation_matrix[job.task.id, cpu_min._identifier] = 1

        return decision


if __name__ == '__main__':
    from simso.generator.generate_schedule import generate_schedule
    from read_csv import read_csv
    import numpy as np
    import os

    execution_times, periods = read_csv("/home/kzagalo/Documents/rInverseGaussian/data/")
    schedule = generate_schedule(execution_times=execution_times[:5], duration=100000,
                                 periods=periods[:5], scheduler='RM', etm='pet')
    premodel = PreEMDF(verbose=True).fit(schedule)
    premodel.save(os.curdir)


