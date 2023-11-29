"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

UNITER finetuning for VQA
"""
import torch
from .model.vqa import UniterForVisualQuestionAnswering
from .misc import set_dropout
from .misc import compute_score_with_logits
from collections import defaultdict
from mixk.models.builder import VQA_MODELS
from ..base_model import BaseModel
import logging

logger = logging.getLogger(__name__)


@VQA_MODELS.register_module()
class UNITER_VQA(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()
        args = kwargs['params']

        # Prepare model
        if args.pretrained_path:
            checkpoint = torch.load(args.pretrained_path)
        else:
            checkpoint = {}

        self.model = UniterForVisualQuestionAnswering.from_pretrained(
            args.model_config,
            checkpoint,
            img_dim=args.img_dim,
            num_answer=args.num_labels,
        )

        # make sure every process has same model parameters in the beginning
        set_dropout(self.model, args.dropout)

    def forward_train(self, data, **kwargs):
        batch = defaultdict(lambda: None, {k: v.cuda() for k, v in data.items()})
        # batch = tuple(t.cuda(device=self.device) for t in data)

        model_output = self.model(batch)

        return model_output

    def forward_test(self, data, **kwargs):
        # excluded data['qid']
        batch = defaultdict(lambda: None, {k: v.cuda() for k, v in data.items() if torch.is_tensor(v)})

        output = self.model(batch)
        scores, target = output['scores'], output['target']

        batch_score = compute_score_with_logits(scores, target).sum()
        batch_size = len(target)  # batch['qids'].size(0)

        model_output = {
            'scores': scores,
            'target': target,
            'batch_score': batch_score,
            'batch_size': batch_size,
        }

        return model_output
