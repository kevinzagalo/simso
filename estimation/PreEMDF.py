from simso.estimation.KmeansInertia import KmeansInertia
from simso.estimation import rInvGaussMixture
import numpy as np
import json
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

    def fit(self, X):
        XX = np.array([x for x in X if all(x)])
        if self.n_tasks_:
            n_tasks = self.n_tasks_
        else:
            _, n_tasks = XX.shape
        self.n_tasks_ = n_tasks

        ## Clustering
        self.kmeans_inertia = KmeansInertia(inertia=self.inertia).fit(XX)
        clusters = self.kmeans_inertia.predict(XX)

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
            for n in range(n_tasks):
                bic_list = []
                igmm_list = []
                for n_components in range(1, self.n_components_max):
                    igmm_list.append(rInvGaussMixture(n_components=n_components).fit(XX[clusters == k, n]))
                    bic_list.append(igmm_list[-1].bic(XX[clusters == k, n]))
                best = np.argmax(bic_list)
                self.mixture_models[k][n] = igmm_list[best]
                self.dmp[k][n] = {}
                for component in range(1, best+1):
                    self.dmp[k][n][component] = self.miss_proba(k, n, component)
        return self

    def miss_proba(self, k, n_task, component):
        q = self.mixture_models[k][n_task].quantile(self.alpha[n_task], component)
        q_norm = (q - self.mixture_models[k][n_task]._means[component]) ** 2 / (self.mixture_models[k][n_task].cv_[0] * q)
        return abs(int(q > self.mixture_models[k][n_task]._means[component]) - chi2.cdf(q_norm, df=1, loc= 0, scale=1))

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def _predict(self, x): # X=[1,2,3,4]
        cluster = self.kmeans_inertia.predict(x)
        task_component = [0] * self.n_tasks_
        for n in range(self.n_tasks_):
            task_component[n] = self.mixture_models[cluster][n].predict(x[n])
        return cluster, task_component

    def predict(self, X): # X=[[1,2,3,4],...,[]]
        if len(np.shape(X)) == 1:
            return self._predict(X)
        elif np.shape(X)[0] == 1:
            return self._predict(X)
        else:
            return [self._predict(x) for x in X]

    def get_parameters(self):
        params = {}
        for k, centroids in enumerate(self.kmeans_inertia.model.cluster_centers_):
            params[k] = {}
            params[k]["centroids"] = list(centroids)
            for n in range(self.n_tasks_):

                params[k][n] = self.mixture_models[k][n].get_parameters()
        params["kmeans_params"] = self.kmeans_inertia.get_parameters()
        params["n_tasks_"] = self.n_tasks_
        params["inertia"] = self.inertia
        return params

    def set_parameters(self, params):
        self.n_tasks_ = params["n_tasks_"]
        self.n_clusters_ = params["kmeans_params"]["n_clusters_"]
        self.kmeans_inertia = KmeansInertia(inertia=self.inertia)
        self.kmeans_inertia.set_parameters(params["kmeans_params"])
        self.mixture_models = {}
        for k in range(self.n_clusters_):
            self.mixture_models[k] = {}
            for n in range(self.n_tasks_):
                self.mixture_models[k][n] = rInvGaussMixture(n_components=params[k][n]["n_components"])
                self.mixture_models[k][n].set_parameters(params[k][n])


    def save(self):
        with open('EMDF.json', 'w') as f:
            json.dump(self.get_parameters(), f)

    def load(self,emdf_json_file):
        self.set_parameters(json.load(open(emdf_json_file)))
        return self


