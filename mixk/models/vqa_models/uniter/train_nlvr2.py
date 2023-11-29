"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

UNITER finetuning for NLVR2
"""
import torch
from .model.nlvr2 import (
    UniterForNlvr2Paired,
    UniterForNlvr2Triplet,
    UniterForNlvr2PairedAttn,
)

from .misc import set_dropout
from collections import defaultdict
from mixk.models.builder import VQA_MODELS
from ..base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

IMG_DIM = 2048


@VQA_MODELS.register_module()
class UNITER_NLVR(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()

        # train_examples = None
        args = kwargs['params']

        if 'paired' in args.model:
            if args.model == 'paired':
                ModelCls = UniterForNlvr2Paired
            elif args.model == 'paired-attn':
                ModelCls = UniterForNlvr2PairedAttn
            else:
                raise ValueError('unrecognized model type')
        elif args.model == 'triplet':
            ModelCls = UniterForNlvr2Triplet
        else:
            raise ValueError('unrecognized model type')

        # Prepare model
        if args.pretrained_path:
            checkpoint = torch.load(args.pretrained_path)
        else:
            checkpoint = {}

        self.model = ModelCls.from_pretrained(
            args.model_config,
            state_dict=checkpoint,
            img_dim=args.img_dim,
        )
        self.model.init_type_embedding()
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

        batch_score = (scores.max(dim=-1, keepdim=False)[1] == target).sum()
        batch_size = len(target)  # batch['qids'].size(0)

        model_output = {
            'scores': scores,
            'target': target,
            'batch_score': batch_score,
            'batch_size': batch_size,
        }

        return model_output
