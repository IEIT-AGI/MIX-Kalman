import logging
from abc import ABCMeta, abstractmethod

import torch

from .evaluator_mixk import DATASET_CONVERTER
from mixk.utils.common_function import byte_tensor_to_object


class BaseDatasetConverter(metaclass=ABCMeta):
    CONVERTER_TO_FUNC = {'evaluator': 'evaluation', 'submitter': 'submit', 'predictor': 'predict'}
    logger = logging.getLogger(__name__)

    def __init__(self, post_process_type: str):
        self._post_process_type = post_process_type

    def convert(self, batch_data, model_outputs, *args, **kwargs):
        try:
            run_func = getattr(self, self.CONVERTER_TO_FUNC[self.post_process_type])
            return run_func(batch_data, model_outputs, *args, **kwargs)
        except KeyError:
            msg = f'The expected type are {self.CONVERTER_TO_FUNC.keys()},but got type is {self.post_process_type}'
            self.logger.info(msg)
            raise KeyError
        except Exception as e:
            self.logger.info(e)
            raise e

    @abstractmethod
    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        pass

    @abstractmethod
    def submit(self, batch_data, model_outputs, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, batch_data, model_outputs, *args, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        return 'base_dataset_converter'

    @property
    def post_process_type(self):
        return self._post_process_type

    @staticmethod
    def list_to_tensor(list_data: list) -> torch.tensor:
        # tensor_size = (len(list_data), list_data[0].shape[1])
        if len(list_data[0].shape) == 0:
            tensor_size = (len(list_data), 1)
        elif len(list_data[0].shape) == 1:
            tensor_size = (len(list_data), list_data[0].shape[0])
        else:
            tensor_size = (len(list_data), list_data[0].shape[1])
        tensor_dtype = list_data[0].dtype
        tensor_data = torch.zeros(size=tensor_size, dtype=tensor_dtype)
        for idx, data in enumerate(list_data):
            tensor_data[idx] = data

        return tensor_data

    @abstractmethod
    # def data_pre_process(self, model_outputs, labels, *args, **kwargs):
    def data_pre_process(self, model_outputs, *args, **kwargs):
        pass


@DATASET_CONVERTER.register_module()
class VQADatasetConverter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)

    def __str__(self):
        return 'vqa_dataset_converter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        from mixk.models.vqa_models.mcan_mix import list2dict
        from mixk.engine.organizer import is_by_iter
        if is_by_iter():
            batch_data = list2dict(batch_data)

        labels = list(batch_data['answers_scores'].split(1))
        q_ids, scores = batch_data['question_id'].split(1), model_outputs['scores'].to('cpu').split(1)
        predictions = list({'question_id': q_id, 'scores': score} for q_id, score in zip(q_ids, scores))
        # predictions, labels = self.data_pre_process(predictions, labels, *args,
        #                                             **kwargs)
        predictions = self.data_pre_process(predictions, *args, **kwargs)
        return predictions, labels

    def submit(self, batch_data, model_outputs, *args, **kwargs):
        # scores, labels = model_outputs['scores'].max(1)
        # q_ids = batch_data['question_id'].detach().numpy()
        # labels = labels.cpu().detach().numpy()
        # q2a = batch_data['quesid2ans']
        # predictions = list({
        # 	                   'question_id': int(qid),
        # 	                   'answer': q2a[l][0]
        #                    } for qid, l in zip(q_ids, labels))
        from mixk.models.vqa_models.mcan_mix import list2dict
        from mixk.engine.organizer import is_by_iter
        if is_by_iter():
            batch_data = list2dict(batch_data)
        q_ids, scores = batch_data['question_id'].split(1), model_outputs['scores'].to('cpu').split(1)
        predictions = list({'question_id': q_id, 'scores': score} for q_id, score in zip(q_ids, scores))
        predictions = self.data_pre_process(predictions, *args, **kwargs)
        # by yinyin q_ids should be question str;q2a should be the answer str
        q2a = batch_data['quesid2ans']
        predictions = list({
            'questionid': str(qid),
            'prediction': str(m[0][l])
        } for qid, l, m in zip(q_ids, predictions, q2a))
        return predictions

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        q_ids = batch_data['question_id'].detach().numpy()
        scores = model_outputs['scores'].cpu().detach().numpy()
        predictions = list({'question_id': int(qid), 'scores': s} for qid, s in zip(q_ids, scores))
        return predictions

    def data_pre_process(self, model_outputs, *args, **kwargs):
        # labels = self.list_to_tensor(labels)
        scores_list = list(model_output['scores'] for model_output in model_outputs)
        scores_tensor = self.list_to_tensor(scores_list)
        predictions = self._get_maxindex(scores_tensor)
        return predictions

    @staticmethod
    def _get_maxindex(output):
        output = VQADatasetConverter._masked_unk_softmax(output, 1, 0)
        output = output.argmax(dim=1)  # argmax
        return output

    @staticmethod
    def _masked_unk_softmax(x, dim, mask_idx):
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y


@DATASET_CONVERTER.register_module()
class VisDialDatasetConverter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)
        self.required_keys = ['sparse_metrics', 'ndcg']

    def __str__(self):
        return 'visdial_datasetconverter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        output = model_outputs['nsp_scores']

        predictions = dict()
        labels = dict()

        def sparse_metrics(predict):
            gt_option_inds = batch_data['gt_option_inds']
            predictions[self.required_keys[0]] = predict
            labels[self.required_keys[0]] = gt_option_inds

        def ndcg_metirc(predict):
            gt_relevance = batch_data['gt_relevance']
            gt_relevance_round_id = batch_data['round_id'].squeeze(1)
            idxs = torch.arange(predict.shape[0])
            output = predict[idxs, gt_relevance_round_id - 1, :]
            predictions[self.required_keys[1]] = output
            labels[self.required_keys[1]] = gt_relevance

        sparse_metrics(output)
        ndcg_metirc(output)

        predictions = [{k: v} for k, v in predictions.items()]
        labels = [{k: v} for k, v in labels.items()]

        return predictions, labels

    def submit(self, batch_data, model_outputs, *args, **kwargs):
        scores, labels = model_outputs['scores'].max(1)
        q_ids = batch_data['question_id'].detach().numpy()
        labels = labels.cpu().detach().numpy()
        q2a = batch_data['quesid2ans']
        predictions = list({'question_id': int(qid), 'answer': q2a[l][0]} for qid, l in zip(q_ids, labels))
        return predictions

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        q_ids = batch_data['question_id'].detach().numpy()
        scores = model_outputs['scores'].cpu().detach().numpy()
        predictions = list({'question_id': int(qid), 'scores': s} for qid, s in zip(q_ids, scores))
        return predictions

    def data_pre_process(self, model_outputs, *args, **kwargs):
        pass


@DATASET_CONVERTER.register_module()
class VCRDatasetConverter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)

    def __str__(self):
        return 'vcr_dataset_converter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        # from mixk.models.vqa_models.mcan_mix import list2dict
        # from mixk.engine.organizer import is_by_iter
        # if is_by_iter():
        #   batch_data = list2dict(batch_data)

        # labels = list(batch_data['answers_scores'].split(1))
        labels = list(model_outputs['target'].split(1))
        # q_ids, scores = batch_data['question_id'].split(
        #     1), model_outputs['scores'].to('cpu').split(1)
        # predictions = list({
        #     'question_id': q_id,
        #     'scores': score
        # } for q_id, score in zip(q_ids, scores))
        predictions = list(model_outputs['scores'].split(1))
        # predictions, labels = self.data_pre_process(predictions, labels, *args,
        #                                             **kwargs)
        predictions = self.data_pre_process(predictions, *args, **kwargs)

        return predictions, labels

    def submit(self, batch_data, model_outputs, *args, **kwargs):
        scores, labels = model_outputs['scores'].max(1)
        q_ids = batch_data['question_id'].detach().numpy()
        labels = labels.cpu().detach().numpy()
        q2a = batch_data['quesid2ans']
        predictions = list({'question_id': int(qid), 'answer': q2a[l][0]} for qid, l in zip(q_ids, labels))
        return predictions

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        q_ids = batch_data['question_id'].detach().numpy()
        scores = model_outputs['scores'].cpu().detach().numpy()
        predictions = list({'question_id': int(qid), 'scores': s} for qid, s in zip(q_ids, scores))
        return predictions

    def data_pre_process(self, model_outputs, *args, **kwargs):
        # labels = self.list_to_tensor(labels)
        # scores_list = list(model_output['scores'] for model_output in model_outputs)
        scores_list = list(model_output for model_output in model_outputs)
        scores_tensor = self.list_to_tensor(scores_list)
        predictions = VCRDatasetConverter._get_accuracy(scores_tensor)
        return predictions

    @staticmethod
    def _get_accuracy(output):
        output = VCRDatasetConverter._masked_unk_softmax(output, 1, 0)
        output = output.argmax(dim=1)  # argmax
        return output

    @staticmethod
    def _masked_unk_softmax(x, dim, mask_idx):
        # x1 = torch.nn.functional.softmax(x, dim=dim)
        x1 = torch.nn.functional.log_softmax(x, dim=dim)
        # x1[:, mask_idx] = 0
        # x1_sum = torch.sum(x1, dim=1, keepdim=True)
        # y = x1 / x1_sum
        y = x1
        return y


@DATASET_CONVERTER.register_module()
class CaptionBleu4Converter(BaseDatasetConverter):

    def __init__(self, post_process_type: str):
        super().__init__(post_process_type=post_process_type)
        self.caption_processor = None
        # self.caption_processor = registry.get("coco_caption_processor")

    def __str__(self):
        return 'CaptionBleu4Converter'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        references = []
        hypotheses = []

        # References
        targets = batch_data.answers
        for j, _ in enumerate(targets):
            img_captions = [self.caption_processor(c)['tokens'] for c in targets[j].tolist()]
            references.append(img_captions)

        # Hypotheses
        if 'captions' in model_outputs:
            scores = model_outputs['captions']
        else:
            scores = torch.max(model_outputs['scores'], dim=-1)[1]
        scores = scores.tolist()
        predictions = []
        for j, _ in enumerate(scores):
            caption = self.caption_processor(scores[j])['tokens']
            predictions.append(caption)
        hypotheses.extend(predictions)

        assert len(references) == len(hypotheses)

        return hypotheses, references

    def submit(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        pass


@DATASET_CONVERTER.register_module()
class TextVQADatasetConvert(BaseDatasetConverter):

    def __init__(self, post_process_type: str, vocab: str):
        super().__init__(post_process_type=post_process_type)
        self.answer_processor_obj = self.init_text_vqa_info_cpler(vocab)

    @staticmethod
    def init_text_vqa_info_cpler(vocab):
        from mixk.data.infocomp.textvqa_infocpler import TextVQAAnswerProcessor
        return TextVQAAnswerProcessor(vocab_file=vocab)

    def __str__(self):
        return 'text_vqa_dataset_convert'

    def evaluation(self, batch_data, model_outputs, *args, **kwargs):
        from mixk.utils.third_party_libs import word_tokenize

        batch_size = batch_data['context_tokens'].size(0)
        predict_answers = model_outputs['scores'].argmax(dim=-1)
        context_tokens = batch_data['context_tokens'].cpu().numpy()
        answers = batch_data['answers'].cpu().numpy()

        answer_space_size = self.answer_processor_obj.get_true_vocab_size()
        predictions = []
        labels = []

        for idx in range(batch_size):
            tokens = byte_tensor_to_object(context_tokens[idx])
            answer_words = []
            for answer_id in predict_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(word_tokenize(tokens[answer_id]))
                else:
                    if answer_id == self.answer_processor_obj.EOS_IDX:
                        break
                    answer_words.append(self.answer_processor_obj.answer_vocab.idx2word(answer_id))

            pred_answer = ' '.join(answer_words).replace(" 's", "'s")
            gt_answers = byte_tensor_to_object(answers[idx])
            predictions.append(pred_answer)
            labels.append(gt_answers)

        return predictions, labels

    def submit(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def predict(self, batch_data, model_outputs, *args, **kwargs):
        pass

    def data_pre_process(self, model_outputs, *args, **kwargs):
        pass
