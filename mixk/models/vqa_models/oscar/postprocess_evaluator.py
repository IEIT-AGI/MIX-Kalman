from mixk.evaluation.dataset_evaluator import BaseDatasetConverter
from mixk.evaluation.metric import BaseMetric
from mixk.evaluation.evaluator_mixk import DATASET_CONVERTER, METRICS
import torch


@DATASET_CONVERTER.register_module()
class OSCAR_DatasetConverter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)

    def __str__(self):
        return 'oscar_dataset_converter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        from mixk.models.vqa_models.mcan_mix import list2dict
        from mixk.engine.organizer import is_by_iter
        if is_by_iter():
            batch_data = list2dict(batch_data)

        return [{'batch_score': model_outputs['batch_score']}], [{'batch_size': model_outputs['batch_size']}]

    # TODO modify
    def submit(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def data_pre_process(self, model_outputs, labels, *args, **kwargs):
        return model_outputs, labels


@METRICS.register_module()
class OSCAR_AccuracyMetric(BaseMetric):
    metric_name = 'oscar_accuracy_metric'

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


@DATASET_CONVERTER.register_module()
class OSCAR_Retrieval_DatasetConverter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)

    def __str__(self):
        return 'oscar_retrieval_dataset_converter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        from mixk.models.vqa_models.mcan_mix import list2dict
        from mixk.engine.organizer import is_by_iter
        if is_by_iter():
            batch_data = list2dict(batch_data)

        eval_result = self.evaluate(batch_data, model_outputs)

        return eval_result

    def evaluate(self, eval_dataset, test_results):
        i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
        rank = [1, 5, 10]
        i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
        logger.info('I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10'.format(i2t_accs[0], i2t_accs[1],
                                                                                   i2t_accs[2]))
        eval_result = {'i2t_retrieval': {'R@1': i2t_accs[0], 'R@5': i2t_accs[1], 'R@10': i2t_accs[2]}}
        if t2i_ranks:
            t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
            logger.info('T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10'.format(
                t2i_accs[0], t2i_accs[1], t2i_accs[2]))
            eval_result['t2i_retrieval'] = {'R@1': t2i_accs[0], 'R@5': t2i_accs[1], 'R@10': t2i_accs[2]}
        return eval_result

    # TODO modify
    def submit(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def data_pre_process(self, model_outputs, labels, *args, **kwargs):
        return model_outputs, labels


# @METRICS.register_module()
# class OSCAR_AccuracyMetric(BaseMetric):
#     metric_name = 'oscar_accuracy_metric'

#     def __init__(self, *args, **kwargs):
#         # cfg = kwargs['cfg']
#         pass

#     def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
#         score = 0.
#         datasize = 0
#         for pred, bsize in zip(predictions, labels):
#             if torch.is_tensor(pred['batch_score']):
#                 batch_score = pred['batch_score'].item()
#             else:
#                 batch_score = pred['batch_score']

#             batch_size = bsize['batch_size']

#             score += batch_score
#             datasize += batch_size

#         return score / datasize
