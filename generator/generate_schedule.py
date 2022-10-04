import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import os
import numpy
from numpy import lcm, sqrt, array
from tqdm import tqdm

from simso.estimation.Diaz import *
from simso.core import Model
from simso.configuration import Configuration
from simso.generator.task_generator import generate_ptask_set
from simso.estimation.Kmeans_inertia import *


def generate_schedule(execution_times, periods, duration=1000):
    configuration = Configuration()
    configuration.verbose = 1
    configuration.alpha=0.1
    configuration.cycles_per_ms = 1
    configuration.duration = duration
    configuration.scheduler_info.clas = "simso.schedulers.RM"
    configuration.etm = 'pet'
    for i, c in enumerate(execution_times):
        configuration.add_task(name="T" + str(i), identifier=int(i + 1), period=periods[i],
                               modes=c[0],
                               proba=c[1], deadline=periods[i], abort_on_miss=True)

    configuration.add_processor(name="CPU 1", identifier=1)
    configuration.check_all()
    model = Model(configuration)
    model.run_model()

    return model.scheduler


if __name__ == '__main__':
    n = 4

    C = pd.read_csv('../../simulations/execution_times.csv', sep=';')
    C = numpy.c_[C[["0"]], C[["1"]]]
    execution_times = []
    for x,y in C:
        x = [int(xx) for xx in x[1:-1].split(',')]
        y = [float(yy) for yy in y[1:-1].split(',')]
        execution_times.append((x,y))
    periods = pd.read_csv('./simulations/periods.csv')[["0"]].values.reshape(1, -1)[0]
    rates = [1/p for p in periods]

    print('Generating schedule...')
    schedule = generate_schedule(execution_times[:n],
                                 periods[:n],
                                 duration=100000)

    print('...schedule generated !')


    #m = array([sum([c0 * c1 for c0, c1 in zip(*c)]) for c in execution_times])
    #U = array([m[i] / periods[i] for i, _ in enumerate(execution_times)])
    #Ubar = cumsum(U)
    #var_s = cumsum([(sum([c0 ** 2 * c1 for c0, c1 in zip(*c)]) - m[i] ** 2)
    #                for i, c in enumerate(execution_times)])
    #var_C = cumsum([sum([c0 ** 2 * c1 for c0, c1 in zip(*c)]) / periods[i]
    #                for i, c in enumerate(execution_times)])
    #gamma = var_C / (1 - Ubar)**2

    #for task in schedule.task_list:
    #    #print('task', task.id, 'U_{} = {}'.format(task.id, Ubar[task.id]))
    #    response_times = []
    #    for j in task._jobs:
    #        if j.response_time is not None:
    #            response_times.append(j.response_time)
    #    pd.DataFrame(response_times).to_csv('./simulations/task_{}.csv'.format(task.id))
    #print('Done.')
    #


