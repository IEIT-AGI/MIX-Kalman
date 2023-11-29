"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

UNITER finetuning for VCR
"""
import torch
from .model.vcr import UniterForVisualCommonsenseReasoning
from .misc import set_dropout
from collections import defaultdict
from mixk.models.builder import VQA_MODELS
from ..base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


@VQA_MODELS.register_module()
class UNITER_VCR(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()

        # train_examples = None
        args = kwargs['params']

        # Prepare model
        if args.pretrained_path and args.checkpoint_from == 'pretrain':
            checkpoint = torch.load(args.pretrained_path)
        else:
            checkpoint = {}

        self.model = UniterForVisualCommonsenseReasoning.from_pretrained(
            args.model_config,
            checkpoint,
            img_dim=args.img_dim,
        )

        self.model.init_type_embedding()
        self.model.init_word_embedding(args.num_special_tokens)

        if args.checkpoint_from == 'vcr_pretrain':
            checkpoint = torch.load(args.pretrained_path)
            state_dict = checkpoint.get('model_state', checkpoint)
            matched_state_dict = {}
            unexpected_keys = set()
            missing_keys = set()
            for name, param in self.model.named_parameters():
                missing_keys.add(name)
            for key, data in state_dict.items():
                if key in missing_keys:
                    matched_state_dict[key] = data
                    missing_keys.remove(key)
                else:
                    unexpected_keys.add(key)
            print('Unexpected_keys:', list(unexpected_keys))
            print('Missing_keys:', list(missing_keys))
            self.model.load_state_dict(matched_state_dict, strict=False)
        del checkpoint

        # make sure every process has same model parameters in the beginning
        set_dropout(self.model, args.dropout)

    def forward_train(self, data, **kwargs):
        batch = defaultdict(lambda: None, {k: v.cuda() for k, v in data.items()})
        # batch = tuple(t.cuda(device=self.device) for t in data)

        output = self.model(batch)

        model_output = {
            'scores': output['scores'],
            'target': output['target'].squeeze(-1),
        }

        return model_output

    def forward_test(self, data, **kwargs):
        # excluded data['qids']
        batch = defaultdict(lambda: None, {k: v.cuda() for k, v in data.items() if torch.is_tensor(v)})

        output = self.model(batch)

        scores = output['scores'][:, 1:]
        qa_targets = batch['qa_targets']
        qar_targets = batch['qar_targets']
        qids = data['qids']
        scores = scores.view(len(qids), -1)
        if scores.shape[1] > 8:
            qar_scores = []
            for batch_id in range(scores.shape[0]):
                answer_ind = qa_targets[batch_id].item()
                qar_index = [4 + answer_ind * 4 + i for i in range(4)]
                qar_scores.append(scores[batch_id, qar_index])
            qar_scores = torch.stack(qar_scores, dim=0)
        else:
            qar_scores = scores[:, 4:]

        curr_qa_score, curr_qar_score, curr_score = self.compute_accuracies(
            scores[:, :4],
            qa_targets,
            qar_scores,
            qar_targets,
        )

        batch_size = len(qids)

        model_output = {
            'qa_batch_score': curr_qa_score,
            'qar_batch_score': curr_qar_score,
            'batch_score': curr_score,
            'batch_size': batch_size,
        }

        return model_output

    def compute_accuracies(self, out_qa, labels_qa, out_qar, labels_qar):
        outputs_qa = out_qa.max(dim=-1)[1]
        outputs_qar = out_qar.max(dim=-1)[1]
        matched_qa = outputs_qa.squeeze() == labels_qa.squeeze()
        matched_qar = outputs_qar.squeeze() == labels_qar.squeeze()
        matched_joined = matched_qa & matched_qar
        n_correct_qa = matched_qa.sum()
        n_correct_qar = matched_qar.sum()
        n_correct_joined = matched_joined.sum()

        return n_correct_qa, n_correct_qar, n_correct_joined
