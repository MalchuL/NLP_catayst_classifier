import torch
import torch.nn as nn


from catalyst.dl import ConfigExperiment

from .dataset import NLPStackOverflowClassificationDataset

torch.backends.cudnn.enabled = False

class Experiment(ConfigExperiment):

    def get_datasets(
        self,
        stage: str,
        train_datapath: str = None,
        test_datapath: str = None,
        infer_datapath: str = None,
        fasttext_model_path = None,
        use_mock=False,
        aggregation='max',
        balance_strategy: str = "upsampling",

    ):

        if stage.startswith('infer'):
            # TODO use_mock=False
            print(use_mock)
            infer = NLPStackOverflowClassificationDataset(infer_datapath, fasttext_model_path, use_sencence_embs=True,
                                                          embs_aggregation=aggregation, use_mock=use_mock)

            datasets = {}

            for dataset, mode in zip(
                    (infer,), ("infer",)
            ):
                datasets[mode] = {'dataset': dataset, 'collate_fn': dataset.get_collate_fn}
        else:
            # TODO use_mock=False
            print(use_mock)
            train = NLPStackOverflowClassificationDataset(train_datapath, fasttext_model_path, use_sencence_embs=True, embs_aggregation=aggregation, use_mock=use_mock)
            test = NLPStackOverflowClassificationDataset(test_datapath, fasttext_model_path, use_sencence_embs=True,
                                                          embs_aggregation=aggregation, use_mock=use_mock)

            datasets  = {}

            for dataset, mode in zip(
                (train, test), ("train", "valid")
            ):

                  datasets[mode] = {'dataset': dataset, 'collate_fn': dataset.get_collate_fn}

        return datasets
