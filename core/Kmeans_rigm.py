# coding=utf-8

from SimPy.Simulation import Simulation
from sklearn.pipeline import Pipeline

from simso.core.Processor import Processor
from simso.core.Task import Task
from simso.core.Timer import Timer
from simso.core.etm import execution_time_models
from simso.core.Logger import Logger
from simso.core.results import Results
from simso.estimation.Modes import Modes
from simso.estimation.Kmeans_inertia import *

from rInverseGaussian.rInvGaussMixture import rInvGaussMixture
#from simso.estimation.rInverseGaussian.rInvGaussMixture import rInvGaussMixture
import matplotlib.pyplot as plt
from numpy import array
import copy
import json

class Kmeans_rigm:

    def __init__(self,
                 alpha=0.1):
        self.alpha=alpha
        self.model={}
        self.n_components_rigm_max = 6

    def delete_0(self, X):
        return array( [list_tasks for list_tasks in X if 0 not in list_tasks] )

    def fit(self, XX):

        kmeans_inertia_ = Kmeans_inertia(alpha=self.alpha)
        XX = kmeans_inertia_.delete_0(array(XX))
        kmeans_inertia_.fit(X=XX)

        list_predicted_classes = kmeans_inertia_.predict(XX)
        list_response_times_by_task = [list(list_response_times_by_task) for list_response_times_by_task in list(numpy.transpose(XX))]
        list_n_components = list(range(1, self.n_components_rigm_max))

        dict_kmeans_rigm_params = dict.fromkeys(list(range(kmeans_inertia_.n_clusters)))

        for unique_class in dict_kmeans_rigm_params:

            dict_kmeans_rigm_params[unique_class] = dict.fromkeys(list(range(len(list_response_times_by_task))))
            for task_n in dict_kmeans_rigm_params[unique_class]:

                list_tasks_class_n_task_n = [list_response_times_by_task[task_n][index] for index in range(len(list_predicted_classes)) if list_predicted_classes[index]==unique_class]

                list_bics_class_n_task_n=[]
                for n_components in list_n_components:
                    r_inv_gauss_mixture_test = rInvGaussMixture(n_components=n_components)
                    r_inv_gauss_mixture_test.fit(X=list_tasks_class_n_task_n)
                    list_bics_class_n_task_n.append(r_inv_gauss_mixture_test.bic(X=list_tasks_class_n_task_n))

                r_inv_gauss_mixture = rInvGaussMixture(n_components=list_n_components[list_bics_class_n_task_n.index(max(list_bics_class_n_task_n))])
                r_inv_gauss_mixture.fit(X=list_tasks_class_n_task_n)
                dict_kmeans_rigm_params[unique_class][task_n]=r_inv_gauss_mixture.get_parameters()

            dict_kmeans_rigm_params[unique_class]["centroids"]=list(kmeans_inertia_.get_centroids()[unique_class])

        self.model = dict_kmeans_rigm_params

    def predict(self, list_lists_tasks):

        centroids = [ self.model[class_n]["centroids"] for class_n in self.model ]


        kmeans = KMeans(n_clusters=len(centroids), n_init=1, random_state=1)

        # pas fini

        if type(list_lists_tasks[0])==int:
            list_lists_tasks=[list_lists_tasks] # [,,,]=>[[,,,]]

        class_predicted = self.kmeans.predict(list_lists_tasks)

        for tasks_i in range(len(list_lists_tasks)):

            for task_n in range(len(list_lists_tasks[tasks_i])):

                response_time=list_lists_tasks[tasks_i][task_n]

                rigm_parameters = self.get_rigm_parameters()[str(task_n)][str(class_predicted[tasks_i])]

                r_inv_gauss_mixture = rInvGaussMixture(n_components=rigm_parameters["n_components"])
                r_inv_gauss_mixture.weights_ = rigm_parameters["weights"]
                r_inv_gauss_mixture.modes_ = rigm_parameters["modes"]
                r_inv_gauss_mixture.cv_ = rigm_parameters["cv"]

                print(response_time)
                print(r_inv_gauss_mixture.predict([response_time]))

