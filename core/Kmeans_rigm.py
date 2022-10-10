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

from simso.estimation.rInverseGaussian.rInvGaussMixture import rInvGaussMixture

import matplotlib.pyplot as plt
from numpy import array
import copy
import json

class Kmeans_rigm:

    def __init__(self,
                 alpha=0.1):
        self.alpha=alpha
        self.model=None
        self.n_components_rigm_max = 6


    def delete_0(self, X):
        return array( [list_tasks for list_tasks in X if 0 not in list_tasks] )


    def write_json_files(self, XX):

        kmeans_inertia_ = Kmeans_inertia(alpha=self.alpha)
        XX = kmeans_inertia_.delete_0(array(XX))

        kmeans_inertia_.fit(X=XX)

        centroids = kmeans_inertia_.get_centroids()
        # centroids n column = task n
        # centroids n row = class n

        dict_response_times_by_task = kmeans_inertia_.dict_response_times_by_task(X=XX)
        # dict = { task_n : { class_n : values ,...},...}

        dict_by_tasks_centroids = kmeans_inertia_.dict_response_times_by_task(X=centroids)
        # dict = { task_n : { class_n : centroid of values ,...},...}

        for task_n in dict_response_times_by_task :
            for class_n in dict_response_times_by_task[task_n]:
                dict_response_times_by_task[task_n][class_n] = {"values":dict_response_times_by_task[task_n][class_n],
                                                                "centroid":dict_by_tasks_centroids[task_n][class_n][0]}

        dict_params = copy.deepcopy(dict_response_times_by_task)

        list_n_components = list(range(1, self.n_components_rigm_max))


        for task_n in dict_response_times_by_task:

            for class_n in dict_response_times_by_task[task_n]:

                task_ = dict_response_times_by_task[task_n][class_n]["values"]
                list_bics = []
                for n_components in list_n_components:
                    r_inv_gauss = rInvGaussMixture(n_components=n_components)
                    r_inv_gauss.fit(X=task_)
                    list_bics.append(r_inv_gauss.bic(X=task_))

                best_n_bytask_byclass = list_n_components[list_bics.index(max(list_bics))]

                r_inv_gauss = rInvGaussMixture(n_components=best_n_bytask_byclass)
                r_inv_gauss.fit(X=task_)
                dict_params[task_n][class_n].update(r_inv_gauss.get_parameters())

                del dict_params[task_n][class_n]['values']

        with open("./simso/core/parameters.json", "w") as outfile:
            json.dump(dict_params, outfile)

    def get_parameters(self):
        self.parameters = json.load(open("./simso/core/parameters.json"))

        return self.parameters


    def fit(self):

        parameters = self.get_parameters()
        # dict paramaters = { task_0 : { kmeans_class_0 : {centroid: ., weights: .,modes: .,cv: .,n_components: .},...},...}

        centroids = []
        for task_n in parameters:

            task_n_class_n_centroids=[]
            for class_n in parameters[task_n]:
                task_n_class_n_centroids.append(parameters[task_n][class_n]["centroid"])

            centroids.append(task_n_class_n_centroids)

        self.kmeans = KMeans(n_clusters=len(centroids), n_init=1, random_state=1)

        self.kmeans.fit(centroids)

        return self.kmeans


    def predict(self, tasks):

        parameters = self.get_parameters()
        class_predicted = self.kmeans.predict(tasks)[0]

        for i in list(range(0, len(tasks[0] ))):
            print(parameters[str(i)][str(class_predicted)])
