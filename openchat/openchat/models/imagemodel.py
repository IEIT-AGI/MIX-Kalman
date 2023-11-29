import torch
from .base_model import BaseModel, MultiModel
import openchat.config as cfg
import cv2
from ..utils import detect_objects_on_single_image as extract_visual_feature

# import sys
# sys.path.insert(0, '/home/datasets/mix_data/openchat/scene_graph_benchmark-main')

# class LxmertBot(BaseModel):
#
#     def __init__(self, env, device, max_context_length):
#         super().__init__('imagemodel', env)
#         self.devices = device.lower()
#         self.max_context_length = max_context_length
#
#         self.eos = '</s><s>'
#         self.lxmert_model = torch.load(cfg.lxmert_weight_path)
#         self.transforms = build_transforms()
#         self.detect_model = torch.load(cfg.detect_weight_path)
#         # self.model.to(device)
#
#     @torch.no_grad()
#     def predict(self, image_id: str, text: str) -> str:
#         torch.cuda.empty_cache()
#         input_ids_list: list = []
#         num_of_stacked_tokens: int = 0
#
#         print(text)
#
#         if image_id not in self.env.histories.keys():
#             self.env.clear(image_id, text)
#
#         user_histories = reversed(self.env.histories[image_id]['user'])
#         bot_histories = reversed(self.env.histories[image_id]['bot'])
#
#         for user, bot in zip(user_histories, bot_histories):
#             user_tokens = self.tokenizer.encode(user, return_tensors='pt')
#             bot_tokens = self.tokenizer.encode(bot, return_tensors='pt')
#             num_of_stacked_tokens += user_tokens.shape[-1] + bot_tokens.shape[-1]
#
#             if num_of_stacked_tokens <= self.max_context_length:
#                 input_ids_list.append(bot_tokens)
#                 input_ids_list.append(user_tokens)
#
#             else:
#                 break
#
#         img_path = cfg.image_path + image_id
#         img = cv2.imread(img_path)
#         dets = detect_objects_on_single_image(self.detect_model, self.transforms, img)
#
#         data = {}
#         data['feats'] = torch.stack([det['features'] for det in dets]).unsqueeze(dim=0)
#         data['boxes'] = torch.stack([torch.tensor(det['rect'], dtype=torch.float32) for det in dets]).unsqueeze(dim=0)
#
#         feats = data['feats'].to('cuda')
#         boxes = data['boxes'].to('cuda')
#         sent = [text]
#
#         output_dict = self.lxmert_model.model(feats, boxes, sent)
#
#         max_score = output_dict['scores'].argmax(dim=-1)
#
#         print(max_score)
#
#         ans = cfg.answer_table[max_score]
#
#         return ans


class LxmertBot(BaseModel, MultiModel):

    def __init__(self, env, device, max_context_length):
        BaseModel.__init__(self, name='imagemodel', env=env)
        MultiModel.__init__(self, scene_graph_weight=cfg.detect_weight_path, cfg=cfg.model_vqa_path.lxmert)

        self.devices = device.lower()
        self.max_context_length = max_context_length
        self.eos = '</s><s>'

    @torch.no_grad()
    def predict(self, image_id: str, text: str) -> str:
        torch.cuda.empty_cache()
        input_ids_list: list = []
        num_of_stacked_tokens: int = 0
        self.logger.info(f'input text:{text}')

        if image_id not in self.env.histories.keys():
            self.env.clear(image_id, text)

        user_histories = reversed(self.env.histories[image_id]['user'])
        bot_histories = reversed(self.env.histories[image_id]['bot'])

        for user, bot in zip(user_histories, bot_histories):
            user_tokens = self.tokenizer.encode(user, return_tensors='pt')
            bot_tokens = self.tokenizer.encode(bot, return_tensors='pt')
            num_of_stacked_tokens += user_tokens.shape[-1] + bot_tokens.shape[-1]

            if num_of_stacked_tokens <= self.max_context_length:
                input_ids_list.append(bot_tokens)
                input_ids_list.append(user_tokens)

            else:
                break

        img_path = cfg.image_path + image_id
        dets = self.run_scene_graph_model(img_path)

        data = {}
        data['feats'] = torch.stack([det['features'] for det in dets]).unsqueeze(dim=0)
        data['boxes'] = torch.stack([torch.tensor(det['rect'], dtype=torch.float32) for det in dets]).unsqueeze(dim=0)

        feats = data['feats'].to('cuda')
        boxes = data['boxes'].to('cuda')
        sent = [text]

        model_output = self.mulitmodal_model.model(feats, boxes, sent)
        max_score = model_output['scores'].argmax(dim=-1)

        self.logger.info(f'max score:{max_score}')
        return self.answer_table[max_score]

    def run_scene_graph_model(self, img_path: str):
        img = cv2.imread(img_path)
        return extract_visual_feature(self.scene_graph_model, self.img_transforms, img)


class VilbertBot(BaseModel, MultiModel):

    def __init__(self, env, device, max_context_length):
        BaseModel.__init__(self, name='imagemodel', env=env)
        MultiModel.__init__(self, scene_graph_weight=cfg.detect_weight_path, cfg=cfg.model_vqa_path.vilbert)

        self.devices = device.lower()
        self.max_context_length = max_context_length
        self.eos = '</s><s>'
        self.padding_index = 0

    @torch.no_grad()
    def predict(self, image_id: str, text: str) -> str:
        torch.cuda.empty_cache()
        input_ids_list: list = []
        num_of_stacked_tokens: int = 0
        self.logger.info(f'input text:{text}')

        if image_id not in self.env.histories.keys():
            self.env.clear(image_id, text)

        user_histories = reversed(self.env.histories[image_id]['user'])
        bot_histories = reversed(self.env.histories[image_id]['bot'])

        for user, bot in zip(user_histories, bot_histories):
            user_tokens = self.tokenizer.encode(user, return_tensors='pt')
            bot_tokens = self.tokenizer.encode(bot, return_tensors='pt')
            num_of_stacked_tokens += user_tokens.shape[-1] + bot_tokens.shape[-1]

            if num_of_stacked_tokens <= self.max_context_length:
                input_ids_list.append(bot_tokens)
                input_ids_list.append(user_tokens)

            else:
                break

        question_token = self.tokenize(text)
        features, num_boxes, img_location = self.visual_info(image_id)

        spatials = img_location
        question = question_token
        image_mask = target = input_mask = segment_ids = co_attention_mask = question_id = torch.Tensor().to(
            device=self.device)
        data = [
            features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id
        ]

        model_output = self.mulitmodal_model(data)
        max_score = model_output['scores'].argmax(dim=-1)

        self.logger.info(f'max score:{max_score}')
        return self.answer_table[max_score]

    def init_token(self, token_cfg):
        from transformers.tokenization_bert import BertTokenizer
        return BertTokenizer.from_pretrained(
            token_cfg.pretrained_model_name_or_path, do_lower_case=token_cfg.do_lower_case)

    def tokenize(self, input_text, max_length=23):
        token = self.tokenizer.encode(input_text)
        token = token[:max_length - 2]
        if len(token) < max_length:
            padding = [self.padding_index] * (max_length - len(token))
            token = token + padding

        return torch.tensor([token], device=self.device)

    def visual_info(self, image_id):
        img_path = cfg.image_path + image_id
        original_visual_feature = self.run_scene_graph_model(img_path)

        img_w, img_h = original_visual_feature.size
        box_features = original_visual_feature.extra_fields['box_features']
        num_boxes = len(original_visual_feature)

        g_feat = torch.sum(box_features, dim=0) / num_boxes
        features = torch.cat([g_feat.view(-1, 2048), box_features], dim=0)
        num_boxes += 1

        boxes = original_visual_feature.bbox
        img_location = torch.zeros((num_boxes - 1, 5), dtype=torch.float32, device=self.device)
        img_location[:, :4] = boxes
        img_location[:, 4] = (img_location[:, 3] - img_location[:, 1]) * (img_location[:, 2] - img_location[:, 0]) / (
            img_w * img_h * 1.0)

        img_location[:, 0] /= img_w
        img_location[:, 2] /= img_w
        img_location[:, 1] /= img_h
        img_location[:, 3] /= img_h
        g_loc = torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32, device=self.device)
        img_location = torch.cat([g_loc.view(-1, 5), img_location], dim=0)

        return features.view(-1, *features.shape), num_boxes, img_location.view(-1, *img_location.shape)

        # g_feat = np.sum(box_features, axis=0) / num_boxes
        # features = np.concatenate([np.expand_dims(g_feat, axis=0), box_features], axis=0)
        # num_boxes += 1
        #
        # boxes = original_visual_feature.bbox
        # img_location = np.zeros((num_boxes - 1, 5), dtype=np.float32)
        # img_location[:, :4] = boxes
        # img_location[:, 4] = (img_location[:, 3] - img_location[:1]) * (img_location[:, 2] - img_location[:, 0]) / (
        #         img_w * img_h * 1.0)
        #
        # img_location[:, 0] /= img_w
        # img_location[:2] /= img_w
        # img_location[:1] /= img_h
        # img_location[:3] /= img_h
        #
        # g_loc = np.array([0, 0, 1, 1, 1])
        # img_location = np.concatenate([np.expand_dims(g_loc, axis=0), img_location], axis=0)
        #
        # return features, num_boxes, img_location


class OscarBot(BaseModel, MultiModel):

    def __init__(self, env, device, max_context_length=128):
        BaseModel.__init__(self, name='imagemodel', env=env)
        MultiModel.__init__(self, scene_graph_weight=cfg.detect_weight_path, cfg=cfg.model_vqa_path.oscar)

        self.devices = device.lower()
        self.max_seq_length = max_context_length
        self.max_img_seq_length = 50
        self.img_feature_dim = 2054
        self.cls_token = '[CLS]',
        self.sep_token = '[SEP]'
        self.pad_token = 0

    @torch.no_grad()
    def predict(self, image_id: str, text: str) -> str:
        torch.cuda.empty_cache()
        input_ids_list: list = []
        num_of_stacked_tokens: int = 0
        self.logger.info(f'input text:{text}')

        if image_id not in self.env.histories.keys():
            self.env.clear(image_id, text)

        user_histories = reversed(self.env.histories[image_id]['user'])
        bot_histories = reversed(self.env.histories[image_id]['bot'])

        for user, bot in zip(user_histories, bot_histories):
            user_tokens = self.tokenizer.encode(user, return_tensors='pt')
            bot_tokens = self.tokenizer.encode(bot, return_tensors='pt')
            num_of_stacked_tokens += user_tokens.shape[-1] + bot_tokens.shape[-1]

            if num_of_stacked_tokens <= self.max_context_length:
                input_ids_list.append(bot_tokens)
                input_ids_list.append(user_tokens)

            else:
                break

        data = self.tensorize(text, image_id)
        data = [d.view(-1, *d.shape) if len(d) else d for d in data]

        model_output = self.mulitmodal_model(data)
        max_score = model_output['scores'].argmax(dim=-1)

        self.logger.info(f'max score:{max_score}')
        return self.answer_table[max_score]

    def init_token(self, token_cfg):
        from transformers.tokenization_bert import BertTokenizer
        return BertTokenizer.from_pretrained(
            token_cfg.pretrained_model_name_or_path, do_lower_case=token_cfg.do_lower_case)

    def tensorize(self, input_text, image_id):

        # text
        sequence_a_segment_id = 0
        cls_token_segment_id = 1
        mask_padding_with_zero = True
        pad_token_segment_id = 0

        tokens = self.tokenizer.encode(input_text)
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[:(self.max_seq_length - 2)]
        tokens = tokens + [self.sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        tokens = [self.cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)

        input_ids = input_ids + ([self.pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        # visual
        img_path = cfg.image_path + image_id
        original_visual_feature = self.run_scene_graph_model(img_path)
        img_w, img_h = original_visual_feature.size
        num_boxes = len(original_visual_feature)
        img_feat = original_visual_feature.extra_fields['box_features'].to('cpu')
        if img_feat.shape[1] < self.img_feature_dim:
            boxes = original_visual_feature.bbox
            img_location = torch.zeros((num_boxes, self.img_feature_dim - img_feat.shape[1]), dtype=img_feat.dtype)
            img_location[:, :4] = boxes
            img_location[:, 4] = (img_location[:, 3] - img_location[:, 1]) / img_h
            img_location[:, 5] = (img_location[:, 2] - img_location[:, 0]) / img_w

            img_location[:, 0] /= img_w
            img_location[:, 2] /= img_w
            img_location[:, 1] /= img_h
            img_location[:, 3] /= img_h

            img_feat = torch.cat([img_feat, img_location], dim=1)

        if img_feat.shape[0] > self.max_img_seq_length:
            img_feat = img_feat[0:self.max_img_seq_length, ]
            if self.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:
            if self.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.max_img_seq_length > 0:
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])

        # other
        label_id = [0]
        score = [0]
        new_scores = self.target_tensor(len(self.answer_table), label_id, score)

        # output
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor([label_id[0]], dtype=torch.long),
            torch.tensor(new_scores, dtype=torch.float),
            img_feat,
            torch.tensor([], dtype=torch.long)  # q_id
        )

    @staticmethod
    def target_tensor(len, labels, scores):
        """create the target by labels and scores."""
        target = [0] * len
        for id, l in enumerate(labels):
            target[l] = scores[id]

        return target


class UNITERBot(BaseModel, MultiModel):

    def __init__(self, env, device, max_context_length):
        BaseModel.__init__(self, name='imagemodel', env=env)
        MultiModel.__init__(self, scene_graph_weight=cfg.detect_weight_path, cfg=cfg.model_vqa_path.uniter)

        self.devices = device.lower()
        self.max_context_length = max_context_length
        self.eos = '</s><s>'
        self.padding_index = 0

    @torch.no_grad()
    def predict(self, image_id: str, text: str) -> str:
        torch.cuda.empty_cache()
        input_ids_list: list = []
        num_of_stacked_tokens: int = 0
        self.logger.info(f'input text:{text}')

        if image_id not in self.env.histories.keys():
            self.env.clear(image_id, text)

        user_histories = reversed(self.env.histories[image_id]['user'])
        bot_histories = reversed(self.env.histories[image_id]['bot'])

        for user, bot in zip(user_histories, bot_histories):
            user_tokens = self.tokenizer.encode(user, return_tensors='pt')
            bot_tokens = self.tokenizer.encode(bot, return_tensors='pt')
            num_of_stacked_tokens += user_tokens.shape[-1] + bot_tokens.shape[-1]

            if num_of_stacked_tokens <= self.max_context_length:
                input_ids_list.append(bot_tokens)
                input_ids_list.append(user_tokens)

            else:
                break

        question_token = self.tokenize(text)
        features, num_boxes, img_location = self.visual_info(image_id)

        spatials = img_location
        question = question_token
        image_mask = target = input_mask = segment_ids = co_attention_mask = question_id = torch.Tensor().to(
            device=self.device)
        data = [
            features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id
        ]

        model_output = self.mulitmodal_model(data)
        max_score = model_output['scores'].argmax(dim=-1)

        self.logger.info(f'max score:{max_score}')
        return self.answer_table[max_score]

    def init_token(self, token_cfg):
        from transformers.tokenization_bert import BertTokenizer
        return BertTokenizer.from_pretrained(
            token_cfg.pretrained_model_name_or_path, do_lower_case=token_cfg.do_lower_case)

    def tokenize(self, input_text, max_length=23):
        token = self.tokenizer.encode(input_text)
        token = token[:max_length - 2]
        if len(token) < max_length:
            padding = [self.padding_index] * (max_length - len(token))
            token = token + padding

        return torch.tensor([token], device=self.device)

    def visual_info(self, image_id):
        img_path = cfg.image_path + image_id
        original_visual_feature = self.run_scene_graph_model(img_path)

        img_w, img_h = original_visual_feature.size
        box_features = original_visual_feature.extra_fields['box_features']
        num_boxes = len(original_visual_feature)

        g_feat = torch.sum(box_features, dim=0) / num_boxes
        features = torch.cat([g_feat.view(-1, 2048), box_features], dim=0)
        num_boxes += 1

        boxes = original_visual_feature.bbox
        img_location = torch.zeros((num_boxes - 1, 5), dtype=torch.float32, device=self.device)
        img_location[:, :4] = boxes
        img_location[:, 4] = (img_location[:, 3] - img_location[:, 1]) * (img_location[:, 2] - img_location[:, 0]) / (
            img_w * img_h * 1.0)

        img_location[:, 0] /= img_w
        img_location[:, 2] /= img_w
        img_location[:, 1] /= img_h
        img_location[:, 3] /= img_h
        g_loc = torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32, device=self.device)
        img_location = torch.cat([g_loc.view(-1, 5), img_location], dim=0)

        return features.view(-1, *features.shape), num_boxes, img_location.view(-1, *img_location.shape)


class VinVLBot(BaseModel, MultiModel):

    def __init__(self, env, device, max_context_length):
        BaseModel.__init__(self, name='imagemodel', env=env)
        MultiModel.__init__(self, scene_graph_weight=cfg.detect_weight_path, cfg=cfg.model_vqa_path.vinvl)

        self.devices = device.lower()
        self.max_context_length = max_context_length
        self.eos = '</s><s>'
        self.padding_index = 0

    @torch.no_grad()
    def predict(self, image_id: str, text: str) -> str:
        torch.cuda.empty_cache()
        input_ids_list: list = []
        num_of_stacked_tokens: int = 0
        self.logger.info(f'input text:{text}')

        if image_id not in self.env.histories.keys():
            self.env.clear(image_id, text)

        user_histories = reversed(self.env.histories[image_id]['user'])
        bot_histories = reversed(self.env.histories[image_id]['bot'])

        for user, bot in zip(user_histories, bot_histories):
            user_tokens = self.tokenizer.encode(user, return_tensors='pt')
            bot_tokens = self.tokenizer.encode(bot, return_tensors='pt')
            num_of_stacked_tokens += user_tokens.shape[-1] + bot_tokens.shape[-1]

            if num_of_stacked_tokens <= self.max_context_length:
                input_ids_list.append(bot_tokens)
                input_ids_list.append(user_tokens)

            else:
                break

        question_token = self.tokenize(text)
        features, num_boxes, img_location = self.visual_info(image_id)

        spatials = img_location
        question = question_token
        image_mask = target = input_mask = segment_ids = co_attention_mask = question_id = torch.Tensor().to(
            device=self.device)
        data = [
            features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id
        ]

        model_output = self.mulitmodal_model(data)
        max_score = model_output['scores'].argmax(dim=-1)

        self.logger.info(f'max score:{max_score}')
        return self.answer_table[max_score]

    def init_token(self, token_cfg):
        from transformers.tokenization_bert import BertTokenizer
        return BertTokenizer.from_pretrained(
            token_cfg.pretrained_model_name_or_path, do_lower_case=token_cfg.do_lower_case)

    def tokenize(self, input_text, max_length=23):
        token = self.tokenizer.encode(input_text)
        token = token[:max_length - 2]
        if len(token) < max_length:
            padding = [self.padding_index] * (max_length - len(token))
            token = token + padding

        return torch.tensor([token], device=self.device)

    def visual_info(self, image_id):
        img_path = cfg.image_path + image_id
        original_visual_feature = self.run_scene_graph_model(img_path)

        img_w, img_h = original_visual_feature.size
        box_features = original_visual_feature.extra_fields['box_features']
        num_boxes = len(original_visual_feature)

        g_feat = torch.sum(box_features, dim=0) / num_boxes
        features = torch.cat([g_feat.view(-1, 2048), box_features], dim=0)
        num_boxes += 1

        boxes = original_visual_feature.bbox
        img_location = torch.zeros((num_boxes - 1, 5), dtype=torch.float32, device=self.device)
        img_location[:, :4] = boxes
        img_location[:, 4] = (img_location[:, 3] - img_location[:, 1]) * (img_location[:, 2] - img_location[:, 0]) / (
            img_w * img_h * 1.0)

        img_location[:, 0] /= img_w
        img_location[:, 2] /= img_w
        img_location[:, 1] /= img_h
        img_location[:, 3] /= img_h
        g_loc = torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32, device=self.device)
        img_location = torch.cat([g_loc.view(-1, 5), img_location], dim=0)

        return features.view(-1, *features.shape), num_boxes, img_location.view(-1, *img_location.shape)
