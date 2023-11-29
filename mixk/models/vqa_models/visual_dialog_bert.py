import torch
from .base_model import BaseModel
from ..builder import VQA_MODELS
from ..visual_dialog_model.vilbert_dialog import BertForMultiModalPreTraining, BertConfig
from mixk.utils.data_utils import sequence_mask


@VQA_MODELS.register_module()
class VisDiaBERT(BaseModel):

    def __init__(self, config):
        super().__init__()
        self.bert_pretrained = BertForMultiModalPreTraining.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            config=BertConfig.from_json_file(config.bert_file_path))
        self.bert_pretrained.train()

        self.sample_size = config.sample_size
        self.dense_options = getattr(config, 'dense_options', self.sample_size)
        self.n_gpus = config.get('n_gpus', 1)

        self.is_dense = config.get('is_dense', False)

    def convert(self, batch, sample_size=None, evaluation=False):

        tokens = batch['tokens']
        segments = batch['segments']
        sep_indices = batch['sep_indices']
        mask = batch['mask']
        hist_len = batch['hist_len']

        # image stuff
        orig_features = batch['image_feat']
        orig_spatials = batch['image_loc']
        orig_image_mask = batch['image_mask']

        tokens = tokens.view(-1, tokens.shape[-1])
        segments = segments.view(-1, segments.shape[-1])
        sep_indices = sep_indices.view(-1, sep_indices.shape[-1])
        mask = mask.view(-1, mask.shape[-1])
        hist_len = hist_len.view(-1)

        features = orig_features.view(-1, orig_features.shape[-2], orig_features.shape[-1])
        spatials = orig_spatials.view(-1, orig_spatials.shape[-2], orig_spatials.shape[-1])
        image_mask = orig_image_mask.view(-1, orig_image_mask.shape[-1])

        if sample_size:
            # subsample a random set
            sample_indices = torch.randperm(hist_len.shape[0])
            sample_indices = sample_indices[:self.sample_size]
        else:
            sample_indices = torch.arange(hist_len.shape[0])

        tokens = tokens[sample_indices, :]
        segments = segments[sample_indices, :]
        sep_indices = sep_indices[sample_indices, :]
        mask = mask[sample_indices, :]
        hist_len = hist_len[sample_indices]

        features = features[sample_indices, :, :]
        spatials = spatials[sample_indices, :, :]
        image_mask = image_mask[sample_indices, :]

        next_sentence_labels = None
        image_target = None
        image_label = None

        if not evaluation:  # self.training
            next_sentence_labels = batch['next_sentence_labels']
            next_sentence_labels = next_sentence_labels.view(-1)
            next_sentence_labels = next_sentence_labels[sample_indices]
            next_sentence_labels = next_sentence_labels.cuda()

            orig_image_target = batch['image_target']
            orig_image_label = batch['image_label']

            image_target = orig_image_target.view(-1, orig_image_target.shape[-2], orig_image_target.shape[-1])
            image_label = orig_image_label.view(-1, orig_image_label.shape[-1])

            image_target = image_target[sample_indices, :, :]
            image_label = image_label[sample_indices, :]

            image_target = image_target.cuda()
            image_label = image_label.cuda()

        tokens = tokens.cuda()
        segments = segments.cuda()
        sep_indices = sep_indices.cuda()
        mask = mask.cuda()
        hist_len = hist_len.cuda()

        features = features.cuda()
        spatials = spatials.cuda()
        image_mask = image_mask.cuda()

        sequence_lengths = torch.gather(sep_indices, 1, hist_len.view(-1, 1)) + 1
        sequence_lengths = sequence_lengths.squeeze(1)
        attention_mask_lm_nsp = sequence_mask(sequence_lengths, max_len=tokens.shape[1])
        sep_len = hist_len + 1
        output_nsp_scores = False
        output_lm_scores = False

        return self._forward(
            tokens,
            features,
            spatials,
            sep_indices=sep_indices,
            sep_len=sep_len,
            token_type_ids=segments,
            masked_lm_labels=mask,
            attention_mask=attention_mask_lm_nsp,
            next_sentence_label=next_sentence_labels,
            output_nsp_scores=output_nsp_scores,
            output_lm_scores=output_lm_scores,
            image_attention_mask=image_mask,
            image_label=image_label,
            image_target=image_target)

    def dense_process_image_data(self, data):
        num_rounds = data['tokens'].shape[1]
        num_samples = data['tokens'].shape[2]
        if self.dense_options is None:
            num_samples = self.dense_options
        orig_features = data['image_feat']
        orig_spatials = data['image_loc']
        orig_image_mask = data['image_mask']
        orig_image_target = data['image_target']
        orig_image_label = data['image_label']

        features = orig_features.unsqueeze(1).unsqueeze(1).expand(orig_features.shape[0], num_rounds, num_samples,
                                                                  orig_features.shape[1],
                                                                  orig_features.shape[2]).contiguous()
        spatials = orig_spatials.unsqueeze(1).unsqueeze(1).expand(orig_spatials.shape[0], num_rounds, num_samples,
                                                                  orig_spatials.shape[1],
                                                                  orig_spatials.shape[2]).contiguous()
        image_label = orig_image_label.unsqueeze(1).unsqueeze(1).expand(orig_image_label.shape[0], num_rounds,
                                                                        num_samples,
                                                                        orig_image_label.shape[1]).contiguous()
        image_mask = orig_image_mask.unsqueeze(1).unsqueeze(1).expand(orig_image_mask.shape[0], num_rounds, num_samples,
                                                                      orig_image_mask.shape[1]).contiguous()
        image_target = orig_image_target.unsqueeze(1).unsqueeze(1).expand(orig_image_target.shape[0], num_rounds,
                                                                          num_samples, orig_image_target.shape[1],
                                                                          orig_image_target.shape[2]).contiguous()

        data['image_feat'] = features.contiguous()
        data['image_loc'] = spatials.contiguous()
        data['image_mask'] = image_mask.contiguous()
        data['image_target'] = image_target.contiguous()
        data['image_label'] = image_label.contiguous()
        return data

    def dense_convert(
        self,
        batch,
        sample_size=None,
        evaluation=False,
        output_nsp_scores=False,
        output_lm_scores=False,
    ):

        gt_option_ind = batch['gt_option'].item()
        all_inds_minus_gt = torch.cat([torch.arange(gt_option_ind), torch.arange(gt_option_ind + 1, 100)], 0)
        all_inds_minus_gt = all_inds_minus_gt[torch.randperm(99)[:self.dense_options - 1]]

        gt = batch['gt_option'].view(-1)
        other_option = all_inds_minus_gt.to(gt.device)
        option_indices = torch.cat([gt, other_option], 0)

        tokens = batch['tokens']
        segments = batch['segments']
        sep_indices = batch['sep_indices']
        mask = batch['mask']
        hist_len = batch['hist_len']
        nsp_labels = batch['next_sentence_labels']

        # select 80 options from the 100 options including the GT option
        tokens = tokens[:, :, option_indices, :]
        segments = segments[:, :, option_indices, :]
        sep_indices = sep_indices[:, :, option_indices, :]
        mask = mask[:, :, option_indices, :]
        hist_len = hist_len[:, :, option_indices]
        nsp_labels = nsp_labels[:, :, option_indices]

        tokens = tokens.view(-1, tokens.shape[-1])
        segments = segments.view(-1, segments.shape[-1])
        sep_indices = sep_indices.view(-1, sep_indices.shape[-1])
        mask = mask.view(-1, mask.shape[-1])
        hist_len = hist_len.view(-1)
        nsp_labels = nsp_labels.view(-1)

        # image stuff
        orig_features = batch['image_feat']
        orig_spatials = batch['image_loc']
        orig_image_mask = batch['image_mask']

        features = orig_features.view(-1, orig_features.shape[-2], orig_features.shape[-1])
        spatials = orig_spatials.view(-1, orig_spatials.shape[-2], orig_spatials.shape[-1])
        image_mask = orig_image_mask.view(-1, orig_image_mask.shape[-1])

        if sample_size:
            # subsample a random set
            sample_indices = torch.randperm(hist_len.shape[0])
            sample_indices = sample_indices[:self.sample_size]
        else:
            sample_indices = torch.arange(hist_len.shape[0])

        tokens = tokens[sample_indices, :]
        segments = segments[sample_indices, :]
        sep_indices = sep_indices[sample_indices, :]
        mask = mask[sample_indices, :]
        hist_len = hist_len[sample_indices]

        features = features[sample_indices, :, :]
        spatials = spatials[sample_indices, :, :]
        image_mask = image_mask[sample_indices, :]

        next_sentence_labels = None
        image_target = None
        image_label = None

        if not evaluation:  # TODO: self.training
            next_sentence_labels = batch['next_sentence_labels']
            next_sentence_labels = next_sentence_labels.view(-1)
            next_sentence_labels = next_sentence_labels[sample_indices]
            next_sentence_labels = next_sentence_labels.cuda()

            orig_image_target = batch['image_target']
            orig_image_label = batch['image_label']

            image_target = orig_image_target.view(-1, orig_image_target.shape[-2], orig_image_target.shape[-1])
            image_label = orig_image_label.view(-1, orig_image_label.shape[-1])

            image_target = image_target[sample_indices, :, :]
            image_label = image_label[sample_indices, :]

            image_target = image_target.cuda()
            image_label = image_label.cuda()

        tokens = tokens.cuda()
        segments = segments.cuda()
        sep_indices = sep_indices.cuda()
        mask = mask.cuda()
        hist_len = hist_len.cuda()

        features = features.cuda()
        spatials = spatials.cuda()
        image_mask = image_mask.cuda()

        sequence_lengths = torch.gather(sep_indices, 1, hist_len.view(-1, 1)) + 1
        sequence_lengths = sequence_lengths.squeeze(1)
        attention_mask_lm_nsp = sequence_mask(sequence_lengths, max_len=tokens.shape[1])
        sep_len = hist_len + 1

        model_output = self._forward(
            tokens,
            features,
            spatials,
            sep_indices=sep_indices,
            sep_len=sep_len,
            token_type_ids=segments,
            masked_lm_labels=mask,
            attention_mask=attention_mask_lm_nsp,
            next_sentence_label=next_sentence_labels,
            output_nsp_scores=output_nsp_scores,
            output_lm_scores=output_lm_scores,
            image_attention_mask=image_mask,
            image_label=image_label,
            image_target=image_target)

        gt_relevance = batch['gt_relevance']
        gt_relevance = gt_relevance[:, option_indices]
        model_output['gt_relevance'] = gt_relevance.cuda()
        model_output['next_sentence_label'] = nsp_labels.cuda()
        return model_output

    def forward_train(self, data, **kwargs):
        if self.is_dense:
            batch = self.dense_process_image_data(data)
            return self.dense_convert(batch, sample_size=None, evaluation=False, output_nsp_scores=True)
        else:
            batch = self.preprocess_data(data)
            return self.convert(batch, sample_size=self.sample_size)

    def forward_test(self, data, **kwargs):
        eval_batch_size = len(data['image_id'])
        num_rounds, num_options = data['tokens'].shape[1], data['tokens'].shape[2]
        # we can fit approximately 500 sequences of length 256 in 8 gpus with 12 GB of memory during inference.
        batch_size = 500 * (self.n_gpus / 8)  # the author uses 8 GPUS for training
        batch_size = min([1, 2, 4, 5, 100, 1000, 200, 8, 10, 40, 50, 500, 20, 25, 250, 125],
                         key=lambda x: abs(x - batch_size) if x <= batch_size else float('inf'))

        batch_data = self.process_test_data(data, eval_batch_size=eval_batch_size, batch_size=batch_size)
        output = []
        for j in range((eval_batch_size * num_rounds * num_options) // batch_size):
            # create chunks of the original batch
            item = {}
            item['tokens'] = batch_data['tokens'][j * batch_size:(j + 1) * batch_size, :]
            item['segments'] = batch_data['segments'][j * batch_size:(j + 1) * batch_size, :]
            item['sep_indices'] = batch_data['sep_indices'][j * batch_size:(j + 1) * batch_size, :]
            item['mask'] = batch_data['mask'][j * batch_size:(j + 1) * batch_size, :]
            item['hist_len'] = batch_data['hist_len'][j * batch_size:(j + 1) * batch_size]

            item['image_feat'] = batch_data['image_feat'][j * batch_size:(j + 1) * batch_size, :, :]
            item['image_loc'] = batch_data['image_loc'][j * batch_size:(j + 1) * batch_size, :, :]
            item['image_mask'] = batch_data['image_mask'][j * batch_size:(j + 1) * batch_size, :]

            model_out = self.convert(item, sample_size=None, evaluation=True)
            # _, _, _, _, nsp_scores = forward(dialog_encoder, item, params, output_nsp_scores=True, evaluation=True)
            # normalize nsp scores
            nsp_scores = model_out['seq_relationship_scores']
            nsp_probs = torch.nn.functional.softmax(nsp_scores, dim=1)
            assert nsp_probs.shape[-1] == 2
            output.append(nsp_probs[:, 0])
        output = torch.cat(output, 0).view(eval_batch_size, num_rounds, num_options)
        return {'nsp_scores': output}

    def preprocess_data(self, data):
        data = self.process_image_data(data)
        return data

    def _forward(self,
                 input_ids,
                 image_feat,
                 image_loc,
                 sep_indices=None,
                 sep_len=None,
                 token_type_ids=None,
                 attention_mask=None,
                 masked_lm_labels=None,
                 next_sentence_label=None,
                 head_mask=None,
                 random_round_indices=None,
                 output_nsp_scores=False,
                 output_lm_scores=False,
                 image_attention_mask=None,
                 image_label=None,
                 image_target=None):

        model_output = self.bert_pretrained(
            input_ids,
            image_feat,
            image_loc,
            sep_indices=sep_indices,
            sep_len=sep_len,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            masked_lm_labels=masked_lm_labels,
            next_sentence_label=next_sentence_label,
            image_attention_mask=image_attention_mask,
            image_label=image_label,
            image_target=image_target)

        return model_output

    @staticmethod
    def process_image_data(data):
        num_rounds = data['tokens'].shape[1]
        num_samples = data['tokens'].shape[2]
        orig_features = data['image_feat']
        orig_spatials = data['image_loc']
        orig_image_mask = data['image_mask']
        orig_image_target = data['image_target']
        orig_image_label = data['image_label']

        features = orig_features.unsqueeze(1).unsqueeze(1).expand(orig_features.shape[0], num_rounds, num_samples,
                                                                  orig_features.shape[1],
                                                                  orig_features.shape[2]).contiguous()
        spatials = orig_spatials.unsqueeze(1).unsqueeze(1).expand(orig_spatials.shape[0], num_rounds, num_samples,
                                                                  orig_spatials.shape[1],
                                                                  orig_spatials.shape[2]).contiguous()
        image_label = orig_image_label.unsqueeze(1).unsqueeze(1).expand(orig_image_label.shape[0], num_rounds,
                                                                        num_samples,
                                                                        orig_image_label.shape[1]).contiguous()
        image_mask = orig_image_mask.unsqueeze(1).unsqueeze(1).expand(orig_image_mask.shape[0], num_rounds, num_samples,
                                                                      orig_image_mask.shape[1]).contiguous()
        image_target = orig_image_target.unsqueeze(1).unsqueeze(1).expand(orig_image_target.shape[0], num_rounds,
                                                                          num_samples, orig_image_target.shape[1],
                                                                          orig_image_target.shape[2]).contiguous()

        data['image_feat'] = features.contiguous()
        data['image_loc'] = spatials.contiguous()
        data['image_mask'] = image_mask.contiguous()
        data['image_target'] = image_target.contiguous()
        data['image_label'] = image_label.contiguous()
        return data

    def process_test_data(self, batch, eval_batch_size, batch_size):
        tokens = batch['tokens']
        num_rounds, num_options = tokens.shape[1], tokens.shape[2]
        tokens = tokens.view(-1, tokens.shape[-1])
        segments = batch['segments']
        segments = segments.view(-1, segments.shape[-1])

        sep_indices = batch['sep_indices']
        sep_indices = sep_indices.view(-1, sep_indices.shape[-1])
        mask = batch['mask']
        mask = mask.view(-1, mask.shape[-1])
        hist_len = batch['hist_len']
        hist_len = hist_len.view(-1)

        # get image features
        features = batch['image_feat']
        spatials = batch['image_loc']
        image_mask = batch['image_mask']
        max_num_regions = features.shape[-2]
        features = features.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions,
                                                             2048).contiguous()
        spatials = spatials.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options, max_num_regions,
                                                             5).contiguous()
        image_mask = image_mask.unsqueeze(1).unsqueeze(1).expand(eval_batch_size, num_rounds, num_options,
                                                                 max_num_regions).contiguous()

        features = features.view(-1, max_num_regions, 2048)
        spatials = spatials.view(-1, max_num_regions, 5)
        image_mask = image_mask.view(-1, max_num_regions)

        assert tokens.shape[0] == segments.shape[0] == sep_indices.shape[0] == mask.shape[0] == \
               hist_len.shape[0] == features.shape[0] == spatials.shape[0] == \
               image_mask.shape[0] == num_rounds * num_options * eval_batch_size

        assert (eval_batch_size * num_rounds * num_options) // batch_size == (eval_batch_size * num_rounds *
                                                                              num_options) / batch_size

        batch_data = {}
        batch_data['tokens'] = tokens
        batch_data['segments'] = segments
        batch_data['sep_indices'] = sep_indices
        batch_data['mask'] = mask
        batch_data['hist_len'] = hist_len

        batch_data['image_feat'] = features
        batch_data['image_loc'] = spatials
        batch_data['image_mask'] = image_mask

        return batch_data
