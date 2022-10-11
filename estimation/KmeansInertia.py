from numpy import array, argmin
from sklearn.cluster import KMeans


class KmeansInertia:

    def __init__(self, alpha=0.1, n_clusters_max=12):
        self.alpha = alpha
        self.model = None
        self.n_clusters_ = n_clusters_max

    def fit(self, X, y=None):
        XX = array([x for x in X if all(x)])
        inertia_0 = (XX - XX.mean(axis=0)**2).sum()
        model_list = []
        inertia_list = []
        for k in range(1, self.n_clusters_):
            kmeans = KMeans(k).fit(XX)
            model_list.append(kmeans)
            inertia_list.append(kmeans.inertia_ / inertia_0 + self.alpha * k)
            model_list.append(kmeans)
        best_k = argmin(inertia_list)
        self.model = model_list[best_k]
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, X):
        return self.model.fit(X).predict(X)




