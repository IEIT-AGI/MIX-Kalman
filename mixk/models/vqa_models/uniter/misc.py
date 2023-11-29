"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Misc utilities
"""
import torch
import logging

logger = logging.getLogger(__name__)


def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p:
                module.p = drop_p
                logger.info(f'{name} set to {drop_p}')


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]  # argmax
    one_hots = torch.zeros(*labels.size(), device=labels.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores
