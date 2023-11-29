from mixk.models.builder import VQA_MODELS
import torch
import copy
'''
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    # BertLayerNorm,
    BertPreTrainedModel,
)
'''
from .lxmert import LXMERTForPretraining
from mixk.models.vqa_models.base_model import BaseModel
from .lxmert_qa_answer_table import load_lxmert_qa
import json
from .lxmert import ClassificationModel


@VQA_MODELS.register_module()
class LXMERT(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()

        args = kwargs['params']
        freeze_base = args['freeze_base']
        training_head_type = args['training_head_type']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if training_head_type == 'pretraining':
            self.model = LXMERTForPretraining(config=args)
            self.forward_train = self.forward_train_pretrain
        else:
            self.model = ClassificationModel(**args)
            pretrained_path = args['pretrained_path']
            if pretrained_path is not None:
                if training_head_type in ['vqa2', 'gqa']:
                    self.label2ans = json.load(open(args.label2ans_path))
                    load_lxmert_qa(pretrained_path, self.model, label2ans=self.label2ans)
                elif training_head_type == 'nlvr2':
                    self.model.lxrt_encoder.load(pretrained_path)

        if freeze_base:
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def forward_train(self, data, **kwargs):
        # ques_id = data['ques_id'].to(self.device)
        feats = data['feats'].to(self.device)
        boxes = data['boxes'].to(self.device)
        sent = data['ques']
        target = data['target'].to(self.device)

        output_dict = self.model(feats, boxes, sent)

        model_output = {
            'scores': output_dict['scores'],
            'target': target,
        }
        return model_output

    def forward_test(self, data):
        model_output = self.forward_train(data)
        return model_output

    def forward_train_pretrain(self, data):
        params = copy.deepcopy(data)
        if params.get('feats') is not None and params.get('image_dim') is not None:
            image_mask = (torch.arange(params['feats'].size(-2)).expand(*params['feats'].size()[:-1]))
            if len(params['image_dim'].size()) < len(image_mask.size()):
                params['image_dim'] = data['image_dim'].unsqueeze(-1)
                assert len(params['image_dim'].size()) == len(image_mask.size())
            image_mask = image_mask < params['image_dim']
            params['visual_attention_mask'] = image_mask.long()
        else:
            params['visual_attention_mask'] = None
        output_dict = self.model(
            input_ids=params['input_ids'].cuda(),
            token_type_ids=params['segment_ids'].cuda(),
            attention_mask=params['input_mask'].cuda(),
            visual_feats=params['feats'].cuda(),
            visual_pos=params['pos'].cuda(),
            visual_attention_mask=params['visual_attention_mask'].cuda()
            if params['visual_attention_mask'] is not None else params['visual_attention_mask'],
        )
        target_dict = {
            'masked_lm_labels': params['lm_label_ids'].cuda(),
            'matched_label': params['is_matched'].cuda(),
            'ans': params['ans'].cuda(),
            'obj_labels': {
                'obj': (params['det_obj_labels'].cuda(), params['det_obj_confs'].cuda()),
                'attr': (params['det_attr_labels'].cuda(), params['det_attr_confs'].cuda()),
                'feat': (params['det_feat'].cuda(), params['det_feat_mask'].cuda()),
            }
        }
        model_output = {'scores': output_dict, 'target': target_dict}
        return model_output

    # def forward_train_pretrain(self, sample_list, **kwargs):
    #
    #     params = self.get_image_and_text_features(sample_list, 'cuda')
    #     if params['visual_feats'] is not None and params['image_dim'] is not None:
    #         device = params['visual_feats'].device
    #         image_mask = (
    #             torch.arange(params['visual_feats'].size(-2)).expand(*params['visual_feats'].size()[:-1]).to(device))
    #         if len(params['image_dim'].size()) < len(image_mask.size()):
    #             params['image_dim'] = params['image_dim'].unsqueeze(-1)
    #             assert len(params['image_dim'].size()) == len(image_mask.size())
    #         image_mask = image_mask < params['image_dim']
    #         params['image_attention_mask'] = image_mask.long()
    #     else:
    #         params['image_attention_mask'] = None
    #     if self.config.training_head_type == 'pretraining':
    #         output_dict = self.model(
    #             input_ids=params['input_ids'],
    #             token_type_ids=params['token_type_ids'],
    #             attention_mask=params['attention_mask'],
    #             visual_feats=params['visual_feats'],
    #             visual_pos=params['pos'],
    #             visual_attention_mask=params['image_attention_mask'],
    #             masked_lm_labels=params['masked_lm_labels'],
    #             masked_image_labels=params['masked_image_labels'],
    #             obj_labels=params['obj_labels'],
    #             matched_label=params['matched_label'],
    #             ans=params['ans'],
    #             num_features=params['max_features'],
    #             name=params['dataset_name'],
    #         )
    #         loss_key = '{}/{}'.format(sample_list.dataset_name, sample_list.dataset_type)
    #         output_dict['losses'] = {}
    #         if 'masked_lm_loss' in output_dict.keys():
    #             output_dict['losses'][loss_key + '/masked_lm_loss'] = output_dict.pop('masked_lm_loss')
    #         if 'matched_loss' in output_dict.keys():
    #             output_dict['losses'][loss_key + '/matched_loss'] = output_dict.pop('matched_loss')
    #         if 'visn_loss' in output_dict.keys():
    #             output_dict['losses'][loss_key + '/visn_loss'] = output_dict.pop('visn_loss')
    #         if 'answer_loss' in output_dict.keys():
    #             output_dict['losses'][loss_key + '/answer_loss'] = output_dict.pop('answer_loss')
    #     else:
    #         output_dict = self.model(
    #             input_ids=params['input_ids'],
    #             token_type_ids=params['token_type_ids'],
    #             attention_mask=params['attention_mask'],
    #             visual_feats=params['visual_feats'],
    #             visual_pos=params['pos'],
    #             visual_attention_mask=params['image_attention_mask'],
    #         )
    #     return output_dict

    def get_image_and_text_features(self, sample_list, device):
        # bert input
        bert_input_ids = sample_list.input_ids
        bert_input_mask = sample_list.input_mask
        bert_input_type_ids = sample_list.segment_ids
        masked_lm_labels = sample_list.lm_label_ids

        # image input
        image_info = getattr(sample_list, 'image_info_0', {})
        image_dim_variable = getattr(image_info, 'max_features', None)
        image_feature_variable = getattr(sample_list, 'image_feature_0', None)
        max_features = torch.tensor(image_feature_variable.shape[1], dtype=torch.int).to(device)
        image_location_variable = getattr(image_info, 'bbox', None)
        image_location_variable = image_location_variable[:, :max_features.item(), :4]

        # aux data
        image_label_variable = getattr(sample_list, 'image_labels', None)
        if image_label_variable is not None:
            image_label_variable = image_label_variable[:, :max_features.item(), None]
            image_label_variable = image_label_variable.unsqueeze(-1).to(device)
        cls_prob = getattr(image_info, 'cls_prob', None)
        if cls_prob is not None:
            cls_prob = torch.tensor(cls_prob)[:, :max_features.item(), None].to(device)
        answers = getattr(sample_list, 'targets', None)
        if answers is None:
            answers = getattr(sample_list, 'answers', None)
        if answers is not None:
            if not isinstance(answers, torch.Tensor):
                answers = torch.tensor(answers)
            answers = answers.to(device)
        is_correct = getattr(sample_list, 'is_correct', None)
        if is_correct is not None:
            if isinstance(is_correct, torch.Tensor):
                is_correct = is_correct.to(device)
            else:
                is_correct = torch.tensor(is_correct).to(device)

        return {
            'input_ids': bert_input_ids,
            'token_type_ids': bert_input_mask,
            'attention_mask': bert_input_type_ids,
            'masked_lm_labels': masked_lm_labels,
            'visual_feats': image_feature_variable,
            'pos': image_location_variable,
            'masked_image_labels': image_label_variable,
            'obj_labels': cls_prob,
            'matched_label': is_correct,
            'ans': answers,
            'image_dim': image_dim_variable,
            'max_features': max_features,
            'dataset_name': str(sample_list.dataset_name),
        }
