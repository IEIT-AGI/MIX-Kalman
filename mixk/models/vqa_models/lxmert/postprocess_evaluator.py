from mixk.evaluation.dataset_evaluator import BaseDatasetConverter
from mixk.evaluation.metric import BaseMetric
from mixk.evaluation.evaluator_mixk import DATASET_CONVERTER, METRICS
from .datasets.lxmert_vqa import VQADataset
from .datasets.lxmert_gqa import GQADataset
from .datasets.lxmert_nlvr2 import NLVR2Dataset
import torch


@DATASET_CONVERTER.register_module()
class LXMERT_VQADatasetConverter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)

    def __str__(self):
        return 'lxmert_vqa_dataset_converter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        from mixk.models.vqa_models.mcan_mix import list2dict
        from mixk.engine.organizer import is_by_iter
        if is_by_iter():
            batch_data = list2dict(batch_data)

        predictions = []
        ques_id = batch_data['ques_id']
        score, label = model_outputs['scores'].max(1)
        for qid, l in zip(ques_id, label.cpu().numpy()):
            if torch.is_tensor(qid):
                predictions.append({qid.item(): l})
            else:
                predictions.append({qid: l})

        return predictions, score

    # TODO modify
    def submit(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def data_pre_process(self, model_outputs, labels, *args, **kwargs):
        return model_outputs, labels


@METRICS.register_module()
class LXMERT_VQAAccuracyMetric(BaseMetric):
    metric_name = 'lxmert_vqa_accuracy_metric'

    def __init__(self, *args, **kwargs):
        cfg = kwargs['cfg']
        self.task = kwargs['task']
        if self.task == 'VQA':
            self.data = VQADataset(cfg)
        elif self.task == 'GQA':
            self.data = GQADataset(cfg)
        elif self.task == 'NLVR':
            self.data = NLVR2Dataset(cfg)

    def calculate(self, predictions: torch.Tensor, labels: torch.Tensor, **kwargs):
        score = 0.
        for pred in predictions:
            [(quesid, l)] = pred.items()
            datum = self.data.id2datum[quesid]
            label = datum['label']

            if self.task in ['VQA', 'GQA']:
                ans = self.data.label2ans[l]
                if ans in label:
                    score += label[ans]
            elif self.task == 'NLVR':
                ans = l
                if ans == label:
                    score += 1

        return score / len(predictions)
