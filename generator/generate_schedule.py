from simso.core import Model
from simso.configuration import Configuration
from simso.generator.task_generator import gen_arrivals_discrete


def generate_schedule(periods, deadlines=None, execution_times=None, duration=1000, scheduler='RM', discard=True,
                      etm='pet', distributions=None, verbose=False, cycles_per_ms=1, n_processors=1, speeds=None,
                      task_type='Periodic'):
    configuration = Configuration()
    configuration.verbose = verbose
    configuration.alpha = 0.1
    configuration.cycles_per_ms = cycles_per_ms
    configuration.duration = duration
    configuration.penalty_preemption = 30
    configuration.penalty_migration = 30
    configuration.scheduler_info.clas = "simso.schedulers."+scheduler
    configuration.etm = etm
    assert execution_times or distributions, "provide either discrete or continuous distribution of execution times"
    if etm == 'pet':
        for i, (c, p) in enumerate(zip(execution_times, periods)):
            if task_type in ('Stationary', 'Sporadic'):
                dates = gen_arrivals_discrete(period=p, duration=duration)
                d = min(p[0])
            else:
                d = p
                dates = None
            configuration.add_task(name="T"+str(i), identifier=int(i+1), task_type=task_type,
                                   list_activation_dates=dates, period=p if task_type == 'Stationary' else d, wcet=max(c[0]),
                                   modes=c[0], proba=c[1], deadline=d, abort_on_miss=discard)
    if etm == 'wcet':
        for i, (c, p) in enumerate(zip(execution_times, periods)):
            configuration.add_task(name="T" + str(i), identifier=int(i + 1), period=min(p[0]), task_type=task_type,
                                   wcet=max(c[0]), deadline=min(deadlines[i][0]), abort_on_miss=discard)
    elif etm == 'continuouset':
        for i, distrib in enumerate(distributions):
            configuration.add_task(name="T"+str(i), identifier=int(i+1), period=periods[i], task_type=task_type,
                                   distribution=distrib, deadline=deadlines[i], abort_on_miss=discard)
    for m in range(n_processors):
        configuration.add_processor(name=f"CPU {m}", identifier=m, migration_overhead=1,
                                    speed=speeds[m] if speeds else 1)
    configuration.check_all()
    model = Model(configuration)
    model.run_model()
    return model
