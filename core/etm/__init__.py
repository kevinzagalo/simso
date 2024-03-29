from .WCET import WCET
from .ACET import ACET
from .pET import pET
from .continuousET import continuousET
from .mixture import mixture
from .CacheModel import CacheModel
from .FixedPenalty import FixedPenalty

execution_time_models = {
    'wcet': WCET,
    'acet': ACET,
    'pet': pET,
    'continuouset': continuousET,
    'mixture': mixture,
    'cache': CacheModel,
    'fixedpenalty': FixedPenalty
}

execution_time_model_names = {
    'WCET': 'wcet',
    'ACET': 'acet',
    'continuousET': 'continuouset',
    'pET': 'pet',
    'mixture': 'mixture',
    'Cache Model': 'cache',
    'Fixed Penalty': 'fixedpenalty'
}
