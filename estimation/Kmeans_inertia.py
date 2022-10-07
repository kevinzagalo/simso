import numpy
from numpy import array
import pandas as pd
from sklearn.cluster import KMeans


class Kmeans_inertia:

    def __init__(self,
                 alpha=0.1):
        self.alpha=alpha
        self.model=None

    def delete_0(self, X):
        return array([list_tasks for list_tasks in X if 0 not in list_tasks])

    def best_n_clusters(self, X, n_clusters_max = 12):
        inertia_o = numpy.square((X - numpy.array(X).mean(axis=0))).sum()
        model_test = KMeans(random_state=0)

        list_inertias = []
        for n_clusters in range(2, n_clusters_max):
            model_test.n_clusters = n_clusters
            model_test.fit(X)

            scaled_inertia = (model_test.inertia_ / inertia_o) + (self.alpha * n_clusters)
            list_inertias.append((n_clusters,
                                 scaled_inertia))

        results = pd.DataFrame(list_inertias, columns=['n_clusters', 'scaled inertia']).set_index('n_clusters')
        self.n_clusters = results.idxmin()[0]


        return self.n_clusters

    def fit(self, X):
        self.model = KMeans(n_clusters=self.best_n_clusters(X))
        self.model.fit(X)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def dict_response_times_by_task(self, X):
        list_predicted_classes = self.model.predict(X)

        dict_response_times_by_task = dict.fromkeys(list(range(0, len(X[0]))))
        # { task_0: {}, ..., task_n: {} }

        list_unique_classes = list(range(self.n_clusters))
        # [0,...,n_clusters]

        for task_n in dict_response_times_by_task:

            dict_response_times_by_task[task_n] = dict.fromkeys(list_unique_classes)

            for unique_class in dict_response_times_by_task[task_n]:

                dict_response_times_by_task[task_n][unique_class] = [X[value_index][task_n] for value_index in [index for index in range(len(list_predicted_classes)) if int(list_predicted_classes[index]) == unique_class] ]

        return dict_response_times_by_task

    def get_centroids(self):
        return self.model.cluster_centers_
