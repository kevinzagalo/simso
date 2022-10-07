# coding=utf-8

from SimPy.Simulation import Simulation

from simso.core.Processor import Processor
from simso.core.Task import Task
from simso.core.Timer import Timer
from simso.core.etm import execution_time_models
from simso.core.Logger import Logger
from simso.core.results import Results
from simso.estimation.Modes import Modes
from simso.estimation.Kmeans_inertia import *
from simso.core.Kmeans_rigm import Kmeans_rigm

from rInverseGaussian.rInvGaussMixture import rInvGaussMixture

import matplotlib.pyplot as plt
from numpy import array
import copy
import json
#, average, lcm, zeros, random


class Model(Simulation):
    """
    Main class for the simulation.
    It instantiate the various components required by the simulation and run it.
    """

    def __init__(self, configuration, callback=None, p=None):
        """
        Args:
            - `callback`: A callback can be specified. This function will be \
                called to report the advance of the simulation (useful for a \
                progression bar).
            - `configuration`: The :class:`configuration \
                <simso.configuration.Configuration>` of the simulation.

        Methods:
        """
        Simulation.__init__(self)
        self._logger = Logger(self)
        task_info_list = configuration.task_info_list
        proc_info_list = configuration.proc_info_list
        self._cycles_per_ms = configuration.cycles_per_ms
        self.p = p
        if 'EDF_MD' in configuration.scheduler_info.clas:
            #if 'RM' in configuration.scheduler_info.train:
            #    configuration.scheduler_info.clas = "simso.schedulers.RM"
            #elif 'US' in configuration.scheduler_info.train:
            configuration.scheduler_info.clas = "simso.schedulers.EDF_US"
            #else:
            #    configuration.scheduler_info.clas = "simso.schedulers.EDF"

            self.p = True

        self.scheduler = configuration.scheduler_info.instantiate(self)
        self.configuration = configuration

        try:
            self._etm = execution_time_models[configuration.etm](
                self, len(proc_info_list)
            )
        except KeyError:
            print("Unknowned Execution Time Model.", configuration.etm)

        self._task_list = []

        for task_info in task_info_list:
            self._task_list.append(Task(self, task_info))

        # Init the processor class. This will in particular reinit the
        # identifiers to 0.
        Processor.init()

        # Initialization of the caches
        for cache in configuration.caches_list:
            cache.init()

        self._processors = []
        for proc_info in proc_info_list:
            proc = Processor(self, proc_info)
            proc.caches = proc_info.caches
            self._processors.append(proc)

        # XXX: too specific.
        self.penalty_preemption = configuration.penalty_preemption
        self.penalty_migration = configuration.penalty_migration

        self._etm.init()

        self._duration = configuration.duration
        self.progress = Timer(self, Model._on_tick, (self,),
                              self.duration // 20 + 1, one_shot=False,
                              in_ms=False)
        self._callback = callback
        self.scheduler.task_list = self._task_list
        self.scheduler.processors = self._processors
        self.results = None
        self.Mmax = 50

    def now_ms(self):
        return float(self.now()) / self._cycles_per_ms

    @property
    def logs(self):
        """
        All the logs from the :class:`Logger <simso.core.Logger.Logger>`.
        """
        return self._logger.logs

    @property
    def logger(self):
        return self._logger

    @property
    def cycles_per_ms(self):
        """
        Number of cycles per milliseconds. A cycle is the internal unit used
        by SimSo. However, the tasks are defined using milliseconds.
        """
        return self._cycles_per_ms

    @property
    def etm(self):
        """
        Execution Time Model
        """
        return self._etm

    @property
    def processors(self):
        """
        List of all the processors.
        """
        return self._processors

    @property
    def task_list(self):
        """
        List of all the tasks.
        """
        return self._task_list

    @property
    def duration(self):
        """
        Duration of the simulation.
        """
        return self._duration

    def _on_tick(self):
        if self._callback:
            self._callback(self.now())

    def run_model(self):

        """ Execute the simulation."""
        self.initialize()
        self.scheduler.init()
        self.scheduler.init_response_time()

        self.progress.start()

        for cpu in self._processors:
            self.activate(cpu, cpu.run())

        for task in self._task_list:
            self.activate(task, task.execute())

        try:
            self.simulate(until=self._duration)
        finally:
            self._etm.update()

            if not self.p and self.now() > 0:
                self.results = Results(self)
                self.results.end()
        """


        plt.hist(array(self.scheduler.idle_times[1:]) - array(self.scheduler.idle_times[:-1]) / self._cycles_per_ms)
        plt.axvline(lcm.reduce([int(t.period / self._cycles_per_ms) for t in self._task_list]))
        plt.title('Distribution of inter-idle times')
        plt.show()

        fig, ax = plt.subplots(figsize=(50, 20))
        ax.plot(self.scheduler.queue)
        ax.scatter(self.scheduler.idle_times, zeros(len(self.scheduler.idle_times)), marker='+', color='red')
        plt.title('trajectory of number of active jobs')
        plt.show()

        lam = average(self.scheduler.queue)

        plt.hist(random.poisson(lam, size=len(self.scheduler.queue)), bin=50)
        plt.show()

        plt.hist(self.scheduler.queue, bin=50)
        plt.title('distribution of number of actives jobs')
        plt.show()

        print(self.scheduler.preemption_classes)

        if self.p:

            X = array(self.scheduler.response_times)
            for i, x in enumerate(X):
                if all(x):
                    X = X[i:, :]
                    break

            fig, ax = plt.subplots(2, len(self.scheduler.task_list), figsize=(25, 8))

            for i, ax_ in enumerate(ax[0]):
                ax_.hist(self.scheduler.task_list[i].response_times, bins=50)
                ax_.set_title('max : {}, mean : {}'.format(max(self.scheduler.task_list[i].response_times),
                                                           average(self.scheduler.task_list[i].response_times)))

            self.configuration.scheduler_info.clas = "simso.schedulers.EDF_MD2"
            self.configuration.scheduler_info.modes = Modes(Mmax=10).fit(X)

            self.__init__(self.configuration, p=True)
            self.initialize()
            self.scheduler.init()
            self.scheduler.init_response_time()

            self.progress.start()

            for cpu in self._processors:
                self.activate(cpu, cpu.run())

            for task in self._task_list:
                self.activate(task, task.execute())

            try:
                self.simulate(until=self._duration)
            finally:
                self._etm.update()

                if self.now() > 0:
                    self.results = Results(self)
                    self.results.end()
                    miss = 0
                    count2 = 0
                    for task in self.scheduler.task_list:
                        print('Task {} : {} jobs, {} miss'.format(task.identifier, len(task._jobs), task.miss_count))
                        miss += task.miss_count
                        count2 += len(task._jobs)
                    print('Total {} jobs, {} miss'.format(count2, miss))
        for i, ax_ in enumerate(ax[1]):
            ax_.hist(self.scheduler.task_list[i].response_times, bins=50)
            ax_.set_title('max : {}, mean : {}'.format(max(self.scheduler.task_list[i].response_times),
                                                       average(self.scheduler.task_list[i].response_times)))

        fig.suptitle('n1 = {}, n2 = {}'.format(count1, count2))
        plt.show()
        """


        if self.configuration.verbose:

            kmeans_rigm=Kmeans_rigm()
            kmeans_rigm.n_components_rigm_max=2
            kmeans_rigm.write_json_files(kmeans_rigm.delete_0(self.scheduler.response_times))

            # ex_taches = [4, 2, 4, 1]
            # print(ex_taches)
            #
            # kmeans_rigm=Kmeans_rigm()
            # kmeans_rigm.fit()
            # print(kmeans_rigm.predict([ex_taches]))
            #
            # rigm=rInvGaussMixture(n_components=3)
            # rigm.weights_ = [0.8177673468882556, 0.06431744120448127, 0.11791521190726449]
            # rigm.modes_=[6.9297181241430215, 15.723227094811218, 10.75620238180887]
            # rigm.cv=[0.1002594776052986, 0.22837929690259934, 0.24102344863770037]
            #
            # print(rigm.fit_predict(ex_taches))


