# Adaptation of the SimSo framework for probabilistic execution times

The original project can be found here: https://github.com/MaximeCheramy/simso/tree/master/simso

## Probabilistic execution times
 
Probabilistic execution times can be simulated now with the ETM's `pET` and `continuousET`. The first one is for discrete distributions, the second one for any continuous distributions available in scipy : https://docs.scipy.org/doc/scipy/reference/stats.html#module-scipy.stats

- For discrete distributions, the inputs to pass to the `configuration.add_task` are `modes for the values of execution times and `proba` for their associated probabilities.
- For continuous distributions, the input is `distribution`.

The function `generator.generate_schedule` returns an instance of the schedule. For example, for discrete distributions one can simulate response times with :

```
from simso.generator.generate_schedule import generate_schedule
execution_times = [([1, 2], [0.5, 0.5]), ([1, 2, 3], [1/3, 1/3, 1/3])]
periods = (4, 6)
schedule = generate_schedule(execution_times=execution_times, periods=periods, etm='pet')
for task in schedule.task_list:
    rt = task.response_times
```

and for continuous function, for example extreme value distributions:

```
from simso.generator.generate_schedule import generate_schedule
from scipy.stats import genextreme as gev

distributions = [gev(loc=10, scale=2), gev(loc=20, scale=1)]  
periods = (20, 36)
schedule = generate_schedule(distributions=distributions, periods=periods, etm='continuouset')
for task in schedule.task_list:
    rt = task.response_times
```

## Modification of the execution time model ACET

In its original version, the `ACET` model generates Gaussian distribution for execution times, but many times it generates negative values, as Gaussian variables is not adapted to model execution times. Now the `ACET` model generated exponential variables, of mean `acet`.

This implementation is not stable !