from simso.estimation.KmeansInertia import KmeansInertia
from simso.estimation import rInvGaussMixture
import numpy
from numpy import array, argmax
import copy
import json
from scipy.optimize import root_scalar

class PreEMDF:

    def __init__(self, inertia=0.1, n_tasks=None, n_components_max=5, alpha=None):
        self.inertia = inertia
        self.alpha = alpha
        self.igmms = {}
        self.kmeans_inertia = None
        self.n_components_max = n_components_max
        self.n_tasks_ = n_tasks
        self.mixture_models = {}
        self.quantiles = {}

    def fit(self, X):
        XX = array([x for x in X if all(x)])
        if self.n_tasks_:
            n_tasks = self.n_tasks_
        else:
            _, n_tasks = XX.shape
        self.n_tasks_ = n_tasks
        self.kmeans_inertia = KmeansInertia(inertia=self.inertia).fit(XX)
        clusters = self.kmeans_inertia.predict(XX)

        k = len(self.kmeans_inertia.model.cluster_centers_)
        self.transitions = numpy.zeros((k, k))
        for (i, j) in zip(clusters[:-1], clusters[1:]):
            self.transitions[i, j] += 1
        for i in range(k):
            self.transitions[i, :] /= self.transitions[i, :].sum()

        for k in set(clusters):
            self.models[k] = {}
            self.quantiles[k] = {}
            for n in range(n_tasks):
                bic_list = []
                igmm_list = []
                for n_components in range(1, self.n_components_max):
                    igmm_list.append(rInvGaussMixture(n_components = n_components).fit(XX[clusters == k, n]))
                    bic_list.append(igmm_list[-1].bic(XX[clusters == k, n]))
                best = np.argmax(bic_list)
                self.models[k][n] = igmm_list[best]
                # self.quantiles[k][n] = igmm_list[best].quantile(self.alpha[n])
        return self

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def fit_predict_quantile(self, X):
        return self.fit(X).predict_quantile(X)

    def _predict(self, X): # X=[1,2,3,4]
        X=[X]
        cluster = self.kmeans_inertia.predict(X)
        task_component = [0] * self.n_tasks_
        for n in range(self.n_tasks_):
            task_component[n] = self.models[cluster][n].predict(X[n])
        return cluster, task_component


    def predict(self, X): # X=[[1,2,3,4],...,[]]
        if array(X).shape == (self.n_tasks_,):
            X = [X]
        return [self._predict(x) for x in X]

    def predict_quantile(self, X):
        cluster, task_component = self.predict(X)
        return self.quantiles[cluster][task_component]

    def get_parameters(self):
        params = {}
        for k, centroids in enumerate(self.kmeans_inertia.model.cluster_centers_):
            params[k] = {}
            params[k]["centroids"] = list(centroids)
            for n in range(self.n_tasks_):

                params[k][n] = self.mixture_models[k][n].get_parameters()
                # rInvGaussMixture_ = rInvGaussMixture(self.mixture_models[k][n])
                # params[k][n] = rInvGaussMixture_.get_parameters()
                # print(self.mixture_models[k][n])

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


