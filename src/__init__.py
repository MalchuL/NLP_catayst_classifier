# flake8: noqa
# isort:skip_file

from catalyst.dl import registry, SupervisedRunner as Runner
from .experiment import Experiment
from .model import NLPClassifierModel
from .metrics import F1MetricCallback
from .callbacks import PredictionCallback


registry.Callback(PredictionCallback)
registry.Callback(F1MetricCallback)
registry.Model(NLPClassifierModel)
