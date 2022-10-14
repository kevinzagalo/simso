from numpy import array, argmin
from sklearn.cluster import KMeans


class KmeansInertia:

    def __init__(self, inertia, n_clusters_max=5, verbose=False):
        self.inertia_ = inertia
        self.model = None
        self.n_clusters_ = n_clusters_max
        self.verbose = verbose

    def fit(self, X, y=None, sample_weight=None):
        XX = array([x for x in X if all(x)])
        inertia_0 = ((XX - XX.mean(axis=0))**2).sum()
        model_list = []
        inertia_list = []
        for k in range(1, self.n_clusters_+1):
            kmeans = KMeans(n_clusters=k).fit(XX, sample_weight=sample_weight)
            inertia_list.append(kmeans.inertia_ / inertia_0 + self.inertia_ * k)
            model_list.append(kmeans)
        if self.verbose:
            import matplotlib.pyplot
            matplotlib.pyplot.plot(range(1, self.n_clusters_+1), inertia_list)
            matplotlib.pyplot.title(self.inertia_)
            matplotlib.pyplot.show()
        best_k = argmin(inertia_list)
        self.model = model_list[best_k]
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def get_parameters(self):
        params = {}
        params["model"] = self.model
        params["model_params"] = self.model.get_params()
        params["inertia_"] = self.inertia_
        params["n_clusters_"] = self.model.n_clusters
        params["cluster_centers_"] = self.model.cluster_centers_
        return params

    def set_parameters(self, params):

        self.model = params["model"]
        self.model.set_params(**params["model_params"])
        self.inertia_ = params["inertia_"]
        self.n_clusters_ = params["n_clusters_"]
        self.model.cluster_centers_ = params["cluster_centers_"]



