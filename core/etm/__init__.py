from .WCET import WCET
from .ACET import ACET
from .pET import pET
from .CacheModel import CacheModel
from .FixedPenalty import FixedPenalty

execution_time_models = {
    'wcet': WCET,
    'acet': ACET,
    'pet': pET,
    'cache': CacheModel,
    'fixedpenalty': FixedPenalty
}

execution_time_model_names = {
    'WCET': 'wcet',
    'ACET': 'acet',
    'pET': 'pet',
    'Cache Model': 'cache',
    'Fixed Penalty': 'fixedpenalty'
}
