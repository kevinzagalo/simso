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
                 alpha=0.01):
        self.alpha=alpha
        self.model=None

    def fit(self,X):
        from sklearn.cluster import KMeans
        n_clusters_max = 12
        list_numbers_clusters = [*range(2, n_clusters_max)]
        alpha=self.alpha
        inertia_o = numpy.square((X - numpy.array(X).mean(axis=0))).sum()

        list_best_ks = []
        model_test=KMeans()
        model_test.random_state = 0
        for k in list_numbers_clusters:
            model_test.n_clusters = k
            model_test.fit(X)

            scaled_inertia = (model_test.inertia_ / inertia_o) + (alpha * k)
            list_best_ks.append((k, scaled_inertia))
        results = pd.DataFrame(list_best_ks, columns=['k', 'Scaled Inertia']).set_index('k')
        best_k = results.idxmin()[0]


        self.model=KMeans(n_clusters=best_k)
        self.list_is_and_inertias=list_best_ks
        self.model.fit(X)

        return self

    def list_is_list_inertias(self):

        list_tuples = self.list_is_and_inertias

        list_is = []
        list_inertias = []
        for tuple in list_tuples:
            list_is.append(tuple[0])
            list_inertias.append(tuple[1])

        return list_is,list_inertias

    def plot_inertia(self):

        lists=self.list_is_list_inertias()
        list_is=lists[0]
        list_inertias=lists[1]

        plt.plot(list_is, list_inertias)
        plt.title(self.alpha)

        plt.show()


    def predict(self,X):
        return self.model.predict(X)


    def barplot_classes(self,X):

        tasks=X
        classes=self.model.predict(tasks)
        plt.hist(classes)
        plt.title(str(len(numpy.unique(classes))))
        plt.show()


    def dict_by_classes(self,X):

        list_jobs=X
        list_classes=self.model.predict(X)

        dict_by_classes = dict.fromkeys(sorted(numpy.unique(list_classes)))

        for unique_class in list(dict_by_classes.keys()):

            dict_by_classes[unique_class] = dict.fromkeys([name_task for name_task in range(0,
                                                                                            len(list_jobs[0]))])

            list_lists_tasks = []
            for numero_task in list(dict_by_classes[unique_class].keys()):

                list_tasks = []
                for job in list_jobs:
                    list_tasks.append(job[numero_task])

                list_lists_tasks.append(list_tasks)

                list_values_at_index = []
                for value_index in [index for index in range(len(list_classes)) if list_classes[index] == unique_class]:
                    list_values_at_index.append(list_lists_tasks[numero_task][value_index])

                dict_by_classes[unique_class][numero_task] = list_values_at_index

        self.dict_by_classes=dict_by_classes
        return self.dict_by_classes

    def print_dict(self):
        dict_by_classes=self.dict_by_classes

        for numero_classe in dict_by_classes:
            for task_name in dict_by_classes[numero_classe]:
                print("class:", numero_classe,
                      " task:", task_name,
                      " : ",
                      dict_by_classes[numero_classe][task_name])




