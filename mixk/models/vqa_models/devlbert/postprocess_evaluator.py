from mixk.evaluation.dataset_evaluator import BaseDatasetConverter
from mixk.evaluation.metric import BaseMetric
from mixk.evaluation.evaluator_mixk import DATASET_CONVERTER, METRICS
import torch


@DATASET_CONVERTER.register_module()
class DEVLBERT_DatasetConverter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)

    def __str__(self):
        return 'vilbert_dataset_converter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        from mixk.models.vqa_models.mcan_mix import list2dict
        from mixk.engine.organizer import is_by_iter
        if is_by_iter():
            batch_data = list2dict(batch_data)

        return [{'batch_score': model_outputs['batch_score']}], [{'batch_size': model_outputs['batch_size']}]

    def submit(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def data_pre_process(self, model_outputs, labels, *args, **kwargs):
        return model_outputs, labels


@METRICS.register_module()
class DEVLBERT_AccuracyMetric(BaseMetric):
    metric_name = 'vilbert_accuracy_metric'

    def __init__(self, *args, **kwargs):
        # cfg = kwargs['cfg']
        pass

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        score = 0.
        datasize = 0
        for pred, bsize in zip(predictions, labels):
            if torch.is_tensor(pred['batch_score']):
                batch_score = pred['batch_score'].item()
            else:
                batch_score = pred['batch_score']

            batch_size = bsize['batch_size']

            score += batch_score
            datasize += batch_size

        return score / datasize
