from mixk.models.builder import VQA_MODELS
import torch
from mixk.models.vqa_models.lxmert.datasets.lxmert_pretrain import convert_example_to_features
import numpy as np
from .lxmert_source.entry import set_visual_config
from .lxmert_source.tokenization import BertTokenizer
from .lxmert_source.modeling import LXRTPretraining

from mixk.models.vqa_models.base_model import BaseModel
from mixk.utils.config import mixkEasyDict


@VQA_MODELS.register_module()
class LXMERT_Pretrain(BaseModel):

    def __init__(self, **kwargs):
        args = kwargs['params']
        super().__init__()
        self.max_seq_length = args.max_seq_length

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Build model
        set_visual_config(args)
        self.model = LXRTPretraining.from_pretrained(
            'bert-base-uncased',
            task_mask_lm=args.task_mask_lm,
            task_obj_predict=args.task_obj_predict,
            task_matched=args.task_matched,
            task_qa=args.task_qa,
            visual_losses=args.visual_losses,
            num_answers=args.num_answers)

        # Weight initialization and loading
        if args.from_scratch:
            self.model.apply(self.model.init_bert_weights)
        if args.load is not None:
            self.load(args.load)
        if args.load_lxmert is not None:
            # Load lxmert would not load the answer head.
            self.load_lxmert(args.load_lxmert)

    def forward_train(self, examples, **kwargs):
        train_features = [
            convert_example_to_features(example, self.max_seq_length, self.tokenizer) for example in examples
        ]

        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()

        # Language Prediction
        lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Prediction
        obj_labels = {}
        for key in ('obj', 'attr', 'feat'):
            visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features])).cuda()
            visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features])).cuda()
            assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
            obj_labels[key] = (visn_labels, visn_mask)

        # Joint Prediction
        matched_labels = torch.tensor([f.is_matched for f in train_features], dtype=torch.long).cuda()
        ans = torch.from_numpy(np.stack([f.ans for f in train_features])).cuda()

        loss, losses, answer_score_logit = self.model(input_ids, segment_ids, input_mask, lm_labels, feats, pos,
                                                      obj_labels, matched_labels, ans)

        output = mixkEasyDict()
        output.loss = loss  # total loss
        output.losses = losses  # every loss
        output.scores = answer_score_logit

        return output
