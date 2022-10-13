from simso.estimation.KmeansInertia import KmeansInertia
from simso.estimation import rInvGaussMixture
from numpy import array, argmax
import copy
import json

class PreEMDF:

    def __init__(self, inertia=0.1, n_tasks=None, n_components_max=5, alpha=None):
        self.inertia = inertia
        self.alpha = alpha
        self.igmms = {}
        self.kmeans = None
        self.n_components_max = n_components_max
        self.n_tasks_ = n_tasks
        self.models = {}
        self.quantiles = {}

    def fit(self, X):
        XX = array([x for x in X if all(x)])
        if self.n_tasks_:
            n_tasks = self.n_tasks_
        else:
            _, n_tasks = X.shape
        self.kmeans = KmeansInertia(alpha=self.inertia).fit(XX)
        clusters = self.kmeans.predict(XX)

        for k in set(clusters):
            self.models[k] = {}
            self.quantiles[k] = {}
            for n in range(n_tasks):
                bic_list = []
                igmm_list = []
                for n_components in range(1, self.n_components_max):
                    igmm_list.append(rInvGaussMixture(n).fit(X[clusters == k, n]))
                    bic_list.append(igmm_list[-1].bic(X[clusters == k, n]))
                best = argmax(bic_list)
                self.models[k][n] = igmm_list[best]
                self.quantiles[k][n] = igmm_list[best].quantile(self.alpha[n])
        return self

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def fit_predict_quantile(self, X):
        return self.fit(X).predict_quantile(X)

    def predict(self, X):
        cluster = self.kmeans.predict(X)
        task_component = [0] * self.n_tasks_
        for n in range(self.n_tasks_):
            task_component[n] = self.models[cluster][n].predict(X[n])
        return cluster, task_component

    def predict_quantile(self, X):
        cluster, task_component = self.predict(X)
        return self.quantiles[cluster][task_component]

    def get_parameters(self):
        params = {}
        for k, centroids in enumerate(self.kmeans.model.cluster_centers_):
            params[k] = {}
            params[k]["centroids"] = list(centroids)
            for n in range(self.n_tasks_):
                params[k][n] = self.models[k][n].get_parameters()

        params["n_tasks_"] = self.n_tasks_
        params["inertia"] = self.inertia
        return params

    def set_parameters(self, params):
        centroids = []
        for k in params:
            centroids.append(params[k]["centroids"])
            for n in range(self.n_tasks_):
                rInvGaussMixture_ = rInvGaussMixture(n_components=params[k][n]["n_components"])
                rInvGaussMixture_.set_parameters(params=params[k][n])
                self.models[k][n] = rInvGaussMixture_
        self.kmeans = KmeansInertia(params["inertia"])
        self.n_tasks_ = params["n_tasks_"]