import numpy
from numpy import lcm, sqrt, array
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
"""
from simso.estimation.Diaz import *
from simso.core import Model
from simso.configuration import Configuration
from simso.generator.task_generator import generate_ptask_set
"""


class Kmeans_inertia:

    def __init__(self,
                 alpha=0.1):
        self.alpha=alpha
        self.model=None

    def delete_0(self, X):
        return array( [list_tasks for list_tasks in X if 0 not in list_tasks] )

    def fit(self, X, n_clusters_max = 12):

        list_n_clusters = [*range(2, n_clusters_max)]
        inertia_o = numpy.square((X - numpy.array(X).mean(axis=0))).sum()

        list_best_ks = []
        model_test = KMeans(random_state = 0)

        for k in list_n_clusters:
            model_test.n_clusters = k
            model_test.fit(X)

            scaled_inertia = (model_test.inertia_ / inertia_o) + (self.alpha * k)
            list_best_ks.append((k, scaled_inertia))

        results = pd.DataFrame(list_best_ks, columns=['k', 'Scaled Inertia']).set_index('k')
        best_k = results.idxmin()[0]

        self.model = KMeans(n_clusters=best_k)
        self.list_is_and_inertias = list_best_ks
        self.model.fit(X)

        return self

    def predict(self, X):
        return self.model.predict(X)


    def dict_by_tasks(self, X):

        list_response_times = X
        list_predicted_classes = self.model.predict(X)

        dict_by_tasks = dict.fromkeys(list(range(0, len(list_response_times[0]))))
        list_unique_classes = [int(unique_class) for unique_class in sorted(numpy.unique(list_predicted_classes))]

        for task_name in dict_by_tasks:

            dict_by_tasks[task_name] = dict.fromkeys(list_unique_classes)

            for unique_class in dict_by_tasks[task_name]:

                list_values = []
                for value_index in [index for index in range(len(list_predicted_classes)) if
                                    int(list_predicted_classes[index]) == unique_class]:
                    list_values.append(list_response_times[value_index][task_name])

                dict_by_tasks[task_name][unique_class] = list_values

        return dict_by_tasks


    def get_centroids(self):
        return self.model.cluster_centers_





