import torch
import torch.nn as nn
from catalyst.data import BalanceClassSampler

from catalyst.dl import ConfigExperiment

from .dataset import SiburClassificationDataset

torch.backends.cudnn.enabled = False

class Experiment(ConfigExperiment):

    def get_datasets(
        self,
        stage: str,
        train_datapath: str = None,
        test_datapath: str = None,
        infer_datapath: str = None,

        balance_strategy: str = "upsampling",

    ):

        if stage.startswith('infer'):
            # TODO use_mock=False
            infer = SiburClassificationDataset(infer_datapath)

            datasets = {}

            for dataset, mode in zip(
                    (infer,), ("infer",)
            ):
                datasets[mode] = {'dataset': dataset, 'collate_fn': dataset.get_collate_fn}
        else:
            # TODO use_mock=False
            train = SiburClassificationDataset(train_datapath)
            test = SiburClassificationDataset(test_datapath)

            datasets  = {}

            for dataset, mode in zip(
                (train, test), ("train", "valid")
            ):

                  datasets[mode] = {'dataset': dataset, 'collate_fn': dataset.get_collate_fn}
                  if mode == 'train':
                      datasets[mode]['sampler'] = BalanceClassSampler(dataset.get_labels(), balance_strategy)

        return datasets
