from __future__ import print_function
import sys
import importlib as imp
import os.path
import importlib
import pkgutil
import inspect

from simso.core.SchedulerEvent import SchedulerBeginScheduleEvent, \
    SchedulerEndScheduleEvent, SchedulerBeginActivateEvent, \
    SchedulerEndActivateEvent, SchedulerBeginTerminateEvent, \
    SchedulerEndTerminateEvent
from SimPy.Simulation import Monitor


class SchedulerInfo(object):
    """
    SchedulerInfo groups the data that characterize a Scheduler (such as the
    scheduling overhead) and do the dynamic loading of the scheduler.
    """
    def __init__(self, clas='', overhead=0, overhead_activate=0,
                 overhead_terminate=0, fields=None, modes=None):
        """
        Args:
            - `name`: Name of the scheduler.
            - `cls`: Class associated to this scheduler.
            - `overhead`: Overhead associated to a scheduling decision.

        Methods:
        """
        self.filename = ''
        self.clas = clas
        self.overhead = overhead
        self.overhead_activate = overhead_activate
        self.overhead_terminate = overhead_terminate
        self.data = {}
        self.fields_types = {}
        self.modes = modes
        self.train = None

        if fields:
            for key, value in fields.items():
                self.data[key] = value[0]
                self.fields_types[key] = value[1]

    def set_fields(self, fields):
        for key, value in fields.items():
            self.data[key] = value[0]
            self.fields_types[key] = value[1]

    def get_cls(self):
        """
        Get the class of this scheduler.
        """
        try:
            clas = None
            if self.clas:
                if type(self.clas) is type:
                    clas = self.clas
                else:
                    name = self.clas.rsplit('.', 1)[1]
                    importlib.import_module(self.clas)
                    clas = getattr(importlib.import_module(self.clas), name)
            elif self.filename:
                path, name = os.path.split(self.filename)
                name = os.path.splitext(name)[0]

                fp, pathname, description = imp.find_module(name, [path])
                if path not in sys.path:
                    sys.path.append(path)
                clas = getattr(imp.load_module(name, fp, pathname,
                                               description), name)
                fp.close()

            return clas
        except Exception as e:
            print("ImportError: ", e)
            if self.clas:
                print("Class: {}".format(self.clas))
            else:
                print("Path: {}".format(self.filename))
            return None

    def instantiate(self, model):
        """
        Instantiate the :class:`Scheduler` class.

        Args:
            - `model`: The :class:`Model <simso.core.Model.Model>` object \
            that is passed to the constructor.
        """
        clas = self.get_cls()
        if clas:
            return clas(model, self)


class Scheduler(object):
    """
    The implementation of a scheduler is done by subclassing this abstract
    class.

    The scheduling events are modeled by method calls which take as arguments
    the :class:`jobs <simso.core.Job.Job>` and the :class:`processors
    <simso.core.Processor.Processor>`.

    The following methods should be redefined in order to interact with the
    simulation:

        - :meth:`init` Called when the simulation is ready. The scheduler \
        logic should be initialized here.
        - :meth:`on_activate` Called upon a job activation.
        - :meth:`on_terminated` Called when a job is terminated.
        - :meth:`schedule` Take the scheduling decision. This method should \
        not be called directly. A call to the :meth:`resched \
        <simso.core.Processor.Processor.resched>` method is required.

    By default, the scheduler can only run on a single processor at the same
    simulation time. It is also possible to override this behavior by
    overriding the :meth:`get_lock` and :meth:`release_lock` methods.
    """

    def __init__(self, sim, scheduler_info, **kwargs):
        """
        Args:

        - `sim`: :class:`Model <simso.core.Model>` instance.
        - `scheduler_info`: A :class:`SchedulerInfo` representing the \
            scheduler.

        Attributes:

        - **sim**: :class:`Model <simso.core.Model.Model>` instance. \
            Useful to get current time with ``sim.now_ms()`` (in ms) or \
            ``sim.now()`` (in cycles).
        - **processors**: List of :class:`processors \
            <simso.core.Processor.Processor>` handled by this scheduler.
        - **task_list**: List of :class:`tasks <simso.core.Task.GenericTask>` \
            handled by this scheduler.

        Methods:
        """
        self.response_times = []
        self.idle_times = [0]
        self.preemption_classes = set()
        self.queue = []
        self.modes = scheduler_info.modes
        self.ARTT = []
        self.backlog_trace = []
        self.sim = sim
        self.processors = []
        self.task_list = []
        self._lock = False
        self.overhead = scheduler_info.overhead
        self.overhead_activate = scheduler_info.overhead_activate
        self.overhead_terminate = scheduler_info.overhead_terminate
        self.data = scheduler_info.data
        self.monitor = Monitor(name="MonitorScheduler", sim=sim)

    def init(self):
        """
        This method is called when the system is ready to run. This method
        should be used in lieu of the __init__ method. This method is
        guaranteed to be called when the simulation starts, after the tasks
        are instantiated
        """
        pass

    def on_activate(self, job):
        """
        This method is called upon a job activation.

        Args:
            - `job`: The activated :class:`job <simso.core.Job.Job>`.
        """
        pass

    def on_terminated(self, job):
        """
        This method is called when a job finish (termination or abortion).

        Args:
            - `job`: The :class:`job <simso.core.Job.Job>` that terminates .
        """
        pass

    def schedule(self, cpu):
        """
        The schedule method must be redefined by the simulated scheduler.
        It takes as argument the cpu on which the scheduler runs.

        Args:
            - `cpu`: The :class:`processor <simso.core.Processor.Processor>` \
            on which the scheduler runs.

        Returns a decision or a list of decisions. A decision is a couple
        (job, cpu).
        """
        raise NotImplementedError("Function schedule to override!")

    def add_task(self, task):
        """
        Add a task to the list of tasks handled by this scheduler.

        Args:
            - `task`: The :class:`task <simso.core.Task.GenericTask>` to add.
        """
        self.task_list.append(task)

    def add_processor(self, cpu):
        """
        Add a processor to the list of processors handled by this scheduler.

        Args:
            - `processor`: The :class:`processor \
            <simso.core.Processor.Processor>` to add.
        """
        self.processors.append(cpu)

    def get_lock(self):
        """
        Implement a lock mechanism. Override it to remove the lock or change
        its behavior.
        """
        if not self._lock:
            self._lock = True
        else:
            return False
        return True

    def release_lock(self):
        """
        Release the lock. Goes in pair with :meth:`get_lock`.
        """
        self._lock = False

    def monitor_begin_schedule(self, cpu):
        self.monitor.observe(SchedulerBeginScheduleEvent(cpu))

    def monitor_end_schedule(self, cpu):
        self.monitor.observe(SchedulerEndScheduleEvent(cpu))

    def monitor_begin_activate(self, cpu):
        self.monitor.observe(SchedulerBeginActivateEvent(cpu))

    def monitor_end_activate(self, cpu):
        self.monitor.observe(SchedulerEndActivateEvent(cpu))

    def monitor_begin_terminate(self, cpu):
        self.monitor.observe(SchedulerBeginTerminateEvent(cpu))

    def monitor_end_terminate(self, cpu):
        self.monitor.observe(SchedulerEndTerminateEvent(cpu))

    @property
    def last_activated_jobs(self):
        if any(c.was_running for c in self.processors):
            return list([int(c.was_running.task.id) for c in self.processors if c.was_running])
        else:
            return None

    def add_response_time(self, job):
        if len(self.ARTT) > 0 and job.response_time:
            rt = job.response_time
            if rt != self.ARTT[job.task.id]:
                self.ARTT[job.task.id] = rt
                self.response_times.append(tuple(self.ARTT))

    def init_response_time(self):
        if self.task_list and len(self.ARTT) == 0:
            self.ARTT = [0] * len(self.task_list)

    def add_backlog(self):
        for task in self.task_list:
            if task.job.is_active():
                self.backlog_trace[task.id] = task.job.ret
            else:
                self.backlog_trace[task.id] = 0


def get_schedulers():
    modules = []

    # Special case when using PyInstaller:
    if getattr(sys, 'frozen', False):
        import pyi_importers
        importer = None
        for obj in sys.meta_path:
            if isinstance(obj, pyi_importers.FrozenImporter):
                importer = obj
                break

        for name in importer.toc:
            if name.startswith('simso.schedulers.'):
                modules.append(name)

    # Normal case:
    else:
        package = importlib.import_module('simso.schedulers')
        for importer, modname, ispkg in pkgutil.walk_packages(
                path=package.__path__,
                prefix=package.__name__ + '.',
                onerror=lambda x: None):
            modules.append(modname)

    for modname in sorted(modules):
        try:
            m = importlib.import_module(modname)
            for name in dir(m):
                c = m.__getattribute__(name)
                if inspect.isclass(c) and issubclass(c, Scheduler):
                    yield modname
                    break
        except (ImportError, SyntaxError):
            print("Import error: ", modname)
