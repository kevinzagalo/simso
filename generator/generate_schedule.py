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


def generate_schedule(periods, execution_times=None,  duration=1000, scheduler='RM',
                      etm='pet', acet=None, distributions=None, verbose=False):
    configuration = Configuration()
    configuration.verbose = verbose
    configuration.alpha = 0.1
    configuration.cycles_per_ms = 1
    configuration.duration = duration
    configuration.scheduler_info.clas = "simso.schedulers."+scheduler
    configuration.etm = etm
    if etm == 'pet':
        for i, c in enumerate(execution_times):
            configuration.add_task(name="T"+str(i), identifier=int(i+1), period=periods[i],
                                   modes=c[0],
                                   proba=c[1], deadline=periods[i], abort_on_miss=True)
    elif etm == 'acet':
        for i, a in enumerate(acet):
            configuration.add_task(name="T"+str(i), identifier=int(i+1), period=periods[i],
                                   acet=a, deadline=periods[i], abort_on_miss=True)
    elif etm == 'continuouset':
        for i, distrib in enumerate(distributions):
            configuration.add_task(name="T"+str(i), identifier=int(i+1), period=periods[i],
                                   distribution=distrib, deadline=periods[i], abort_on_miss=True)

    configuration.add_processor(name="CPU 1", identifier=1)
    configuration.check_all()
    model = Model(configuration)
    print('Generating schedule...')
    model.run_model()
    print('Done !')
    return model.scheduler
