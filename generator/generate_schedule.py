from simso.core import Model
from simso.configuration import Configuration
from simsogui.Gantt import Gantt

def generate_schedule(periods, deadlines=None, execution_times=None,  duration=1000, scheduler='RM', discard=True,
                      etm='pet', acet=None, distributions=None, verbose=False, cycles_per_ms=1, n_processors=1, speeds=None):
    configuration = Configuration()
    configuration.verbose = verbose
    configuration.alpha = 0.1
    configuration.cycles_per_ms = cycles_per_ms
    configuration.duration = duration
    configuration.penalty_preemption = 1
    configuration.penalty_migration = 1
    configuration.scheduler_info.clas = "simso.schedulers."+scheduler
    configuration.etm = etm
    if deadlines is None:
        deadlines = periods
    assert execution_times or distributions, "provide either discrete or continuous distribution of execution times"
    if etm == 'pet':
        for i, c in enumerate(execution_times):
            configuration.add_task(name="T"+str(i), identifier=int(i+1), period=periods[i],
                                   modes=c[0],
                                   proba=c[1], deadline=deadlines[i], abort_on_miss=discard)
    elif etm == 'acet':
        for i, a in enumerate(acet):
            configuration.add_task(name="T"+str(i), identifier=int(i+1), period=periods[i],
                                   acet=a, deadline=deadlines[i], abort_on_miss=discard)
    elif etm == 'continuouset':
        for i, distrib in enumerate(distributions):
            configuration.add_task(name="T"+str(i), identifier=int(i+1), period=periods[i],
                                   distribution=distrib, deadline=deadlines[i], abort_on_miss=discard)

    for m in range(n_processors):
        configuration.add_processor(name=f"CPU {m}", identifier=m, speed=speeds[m] if speeds else 1)
    configuration.check_all()
    model = Model(configuration)
    model.run_model()
    return model
