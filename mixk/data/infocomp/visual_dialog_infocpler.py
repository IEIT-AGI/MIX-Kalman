# from ..utils.tokenization import BertTokenizer
from transformers.tokenization_bert import BertTokenizer
from ..utils.data_utils import encode_input
import random
import torch
from ..vqadata.stream import ItemFeature
from mixk.utils.config import mixkEasyDict


class VisDiaInfoCpler:
    RUN_FUNC_BERT = {
        'train': 'train_dataset_bert_info',
        'val': 'val_dataset_bert_info',
        'test': 'test_dataset_bert_info',
    }

    def __init__(self, cfg):
        self.has_bert = cfg.get('has_bert', False)
        self.tokenizer_path = cfg.tokenizer.path
        self.num_options = cfg.num_options
        self.num_negative_samples = cfg.get('num_negative_samples', 8)
        self.visual_dialog_tot_rounds = cfg.visual_dialog_tot_rounds
        self.max_sequence_len = cfg.get('max_sequence_len', 256)
        self.mask_probability = cfg.get('mask_probability', 0.15)
        self.visdial_tot_rounds = cfg.get('visdial_tot_rounds', 11)
        self.tokenizer_path = cfg.get('tokenizer', 'bert-base-uncased')
        if isinstance(self.tokenizer_path, mixkEasyDict):
            self.tokenizer_path = self.tokenizer_path.path

        self.init_tokens()

    def init_tokens(self):
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=self.tokenizer_path)
        tokens = ['[CLS]', '[MASK]', '[SEP]']
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        self.CLS = indexed_tokens[0]
        self.MASK = indexed_tokens[1]
        self.SEP = indexed_tokens[2]

    def complete_info(self, item_feature, split='train'):
        return self.complete_bert_info(item_feature,
                                       split) if self.has_bert else self.complete_normal_info(item_feature)

    def complete_normal_info(self, item_feature):
        pass

    def complete_bert_info(self, item_feature, split):
        func = getattr(self, self.RUN_FUNC_BERT[split])
        return func(item_feature)

    def generate_train_language_info(self, utterances, utterances_random):
        tokens_all_list = []
        mask_all_list = []
        segments_all_list = []
        sep_indices_all_list = []
        next_labels_all_list = []
        hist_len_all_list = []

        for idx, ut in enumerate(utterances):
            tokens_all = []
            mask_all = []
            segments_all = []
            sep_indices_all = []
            next_labels_all = []
            hist_len_all = []

            ut, start_segment = self.prune_rounds(ut, self.visual_dialog_tot_rounds)
            tokens, segments, sep_indices, mask = encode_input(
                ut,
                start_segment,
                self.CLS,
                self.SEP,
                self.MASK,
                max_seq_len=self.max_sequence_len,
                mask_prob=self.mask_probability)
            tokens_all.append(tokens)
            mask_all.append(mask)
            sep_indices_all.append(sep_indices)
            next_labels_all.append(torch.LongTensor([0]))
            segments_all.append(segments)
            hist_len_all.append(torch.LongTensor([len(ut) - 1]))
            negative_samples = utterances_random[idx]

            for context_random in negative_samples:
                context_random, start_segment = self.prune_rounds(context_random, self.visual_dialog_tot_rounds)
                # print("{}: {}".format(j, tokens2str(context_random)))
                tokens_random, segments_random, sep_indices_random, mask_random = encode_input(
                    context_random,
                    start_segment,
                    self.CLS,
                    self.SEP,
                    self.MASK,
                    max_seq_len=self.max_sequence_len,
                    mask_prob=self.mask_probability)
                tokens_all.append(tokens_random)
                mask_all.append(mask_random)
                sep_indices_all.append(sep_indices_random)
                next_labels_all.append(torch.LongTensor([1]))
                segments_all.append(segments_random)
                hist_len_all.append(torch.LongTensor([len(context_random) - 1]))

            tokens_all_list.append(torch.cat(tokens_all, 0).unsqueeze(0))
            mask_all_list.append(torch.cat(mask_all, 0).unsqueeze(0))
            segments_all_list.append(torch.cat(segments_all, 0).unsqueeze(0))
            sep_indices_all_list.append(torch.cat(sep_indices_all, 0).unsqueeze(0))
            next_labels_all_list.append(torch.cat(next_labels_all, 0).unsqueeze(0))
            hist_len_all_list.append(torch.cat(hist_len_all, 0).unsqueeze(0))

        tokens_all_list = torch.cat(tokens_all_list, 0)
        mask_all_list = torch.cat(mask_all_list, 0)
        segments_all_list = torch.cat(segments_all_list, 0)
        sep_indices_all_list = torch.cat(sep_indices_all_list, 0)
        next_labels_all_list = torch.cat(next_labels_all_list, 0)
        hist_len_all_list = torch.cat(hist_len_all_list, 0)

        language_info = {}
        language_info['tokens'] = tokens_all_list
        language_info['segments'] = segments_all_list
        language_info['sep_indices'] = sep_indices_all_list
        language_info['mask'] = mask_all_list
        language_info['next_sentence_labels'] = next_labels_all_list
        language_info['hist_len'] = hist_len_all_list

        return language_info

    def generate_train_samples(self, item_feature):  # build positvie utterances and negative utterances
        utterances = []
        utterances_random = []
        caption_token = self.tokenizer.encode(item_feature['caption'])
        utterances.append([caption_token])
        utterances_random.append([caption_token])
        token_len = len(caption_token) + 2

        for dl in item_feature['dialog']:
            curr_utterance = utterances[-1].copy()
            curr_utterance_random = utterances[-1].copy()

            q_token = self.tokenizer.encode(dl['question'])
            a_token = self.tokenizer.encode(dl['answer'])

            curr_utterance.append(q_token)
            curr_utterance.append(a_token)

            q_token_len = len(q_token)
            a_token_len = len(a_token)

            token_len += q_token_len + a_token_len + 2  # the question sep token and  answer sep token

            curr_utterance_random.append(q_token)
            utterances.append(curr_utterance)

            gt_idx = dl['gt_index']
            negative_samples = []
            answer_options_num = len(dl['answer_options'])

            for _ in range(self.num_negative_samples):
                all_options_idx = list(range(answer_options_num))
                all_options_idx.remove(gt_idx)
                all_options_idx = all_options_idx[:self.num_options - 1]
                random_utterance_token = None

                random_idx = None
                while len(all_options_idx):
                    random_idx = random.choice(all_options_idx)
                    random_utterance_token = self.tokenizer.encode(dl['answer_options'][random_idx])
                    if self.max_sequence_len >= (token_len + len(random_utterance_token) + 1):
                        break
                    else:
                        all_options_idx.remove(random_idx)

                if len(all_options_idx) == 0:
                    random_utterance_token = random_utterance_token[:a_token_len]

                tmp = curr_utterance_random.copy()
                tmp.append(random_utterance_token)
                negative_samples.append(tmp)

            utterances_random.append(negative_samples)

        return utterances, utterances_random

    def generate_val_samples(self, item_feature):  # build positvie utterances and negative utterances
        utterances = []
        gt_relevance = None
        gt_option_inds = []
        options_all = []
        caption_token = self.tokenizer.encode(item_feature['caption'])
        utterances.append([caption_token])
        num_options = self.num_options

        for idx, utterance in enumerate(item_feature['dialog']):
            cur_rnd_utterance = utterances[-1].copy()
            cur_rnd_utterance.append(self.tokenizer.encode(utterance['question']))
            # current round
            gt_option_ind = utterance['gt_index']
            option_inds = []
            option_inds.append(gt_option_ind)
            all_inds = list(range(100))
            all_inds.remove(gt_option_ind)
            all_inds = all_inds[:(num_options - 1)]
            option_inds.extend(all_inds)
            gt_option_inds.append(0)
            cur_rnd_options = []
            answer_options = [utterance['answer_options'][k] for k in option_inds]
            assert len(answer_options) == len(option_inds) == num_options
            assert answer_options[0] == utterance['answer']

            if idx == item_feature['round_id'] - 1:
                gt_relevance = torch.Tensor(item_feature['gt_relevance'])
                # shuffle based on new indices
                gt_relevance = gt_relevance[torch.LongTensor(option_inds)]
            for a_op in answer_options:
                cur_rnd_cur_option = cur_rnd_utterance.copy()
                cur_rnd_cur_option.append(self.tokenizer.encode(a_op))
                cur_rnd_options.append(cur_rnd_cur_option)
            cur_rnd_utterance.append(self.tokenizer.encode(utterance['answer']))
            utterances.append(cur_rnd_utterance)
            options_all.append(cur_rnd_options)

        return options_all, gt_relevance, gt_option_inds

    def generate_test_samples(self, item_feature):
        options_all = []
        caption_token = self.tokenizer.encode(item_feature['caption'])
        cur_rnd_utterance = [caption_token]
        dialog_len = len(item_feature['dialog'])
        for idx, dl in enumerate(item_feature['dialog']):
            q_token = self.tokenizer.encode(dl['question'])
            cur_rnd_utterance.append(q_token)
            if idx != dialog_len - 1:
                cur_rnd_utterance.append(self.tokenizer.encode(dl['answer']))

        for answer in item_feature['dialog'][-1]['answer_options']:
            cur_option = cur_rnd_utterance.copy()
            cur_option.append(self.tokenizer.encode(answer))
            options_all.append(cur_option)

        return options_all

    def generate_val_language_info(self, options_all):
        tokens_all = []
        mask_all = []
        segments_all = []
        sep_indices_all = []
        hist_len_all = []
        for rnd, cur_rnd_options in enumerate(options_all):

            tokens_all_rnd = []
            mask_all_rnd = []
            segments_all_rnd = []
            sep_indices_all_rnd = []
            hist_len_all_rnd = []

            for cur_rnd_option in cur_rnd_options:
                cur_rnd_option, start_segment = self.prune_rounds(cur_rnd_option, self.visdial_tot_rounds)
                tokens, segments, sep_indices, mask = encode_input(
                    cur_rnd_option,
                    start_segment,
                    self.CLS,
                    self.SEP,
                    self.MASK,
                    max_seq_len=self.max_sequence_len,
                    mask_prob=self.mask_probability)

                tokens_all_rnd.append(tokens)
                mask_all_rnd.append(mask)
                segments_all_rnd.append(segments)
                sep_indices_all_rnd.append(sep_indices)
                hist_len_all_rnd.append(torch.LongTensor([len(cur_rnd_option) - 1]))

            tokens_all.append(torch.cat(tokens_all_rnd, 0).unsqueeze(0))
            mask_all.append(torch.cat(mask_all_rnd, 0).unsqueeze(0))
            segments_all.append(torch.cat(segments_all_rnd, 0).unsqueeze(0))
            sep_indices_all.append(torch.cat(sep_indices_all_rnd, 0).unsqueeze(0))
            hist_len_all.append(torch.cat(hist_len_all_rnd, 0).unsqueeze(0))

        tokens_all = torch.cat(tokens_all, 0)
        mask_all = torch.cat(mask_all, 0)
        segments_all = torch.cat(segments_all, 0)
        sep_indices_all = torch.cat(sep_indices_all, 0)
        hist_len_all = torch.cat(hist_len_all, 0)

        item = {}
        item['tokens'] = tokens_all
        item['segments'] = segments_all
        item['sep_indices'] = sep_indices_all
        item['mask'] = mask_all
        item['hist_len'] = hist_len_all

        return item

    def generate_test_language_info(self, options_all):
        tokens_all = []
        mask_all = []
        segments_all = []
        sep_indices_all = []
        hist_len_all = []

        for option in options_all:
            option, start_segment = self.pruneRounds(option, self.visdial_tot_rounds)
            # print("option: {} {}".format(j, tokens2str(option)))
            tokens, segments, sep_indices, mask = encode_input(
                option, start_segment, self.CLS, self.SEP, self.MASK, max_seq_len=self.max_sequence_len, mask_prob=0)

            tokens_all.append(tokens)
            mask_all.append(mask)
            segments_all.append(segments)
            sep_indices_all.append(sep_indices)
            hist_len_all.append(torch.LongTensor([len(option) - 1]))

        tokens_all = torch.cat(tokens_all, 0)
        mask_all = torch.cat(mask_all, 0)
        segments_all = torch.cat(segments_all, 0)
        sep_indices_all = torch.cat(sep_indices_all, 0)
        hist_len_all = torch.cat(hist_len_all, 0)

        language_info = {}
        language_info['tokens'] = tokens_all.unsqueeze(0)
        language_info['segments'] = segments_all.unsqueeze(0)
        language_info['sep_indices'] = sep_indices_all.unsqueeze(0)
        language_info['mask'] = mask_all.unsqueeze(0)
        language_info['hist_len'] = hist_len_all.unsqueeze(0)

        return language_info

    def tokens2str(self, seq):
        dialog_sequence = ''
        for sentence in seq:
            for word in sentence:
                dialog_sequence += self.tokenizer._convert_id_to_token(word) + ' '
                dialog_sequence += ' </end> '
        dialog_sequence = dialog_sequence.encode('utf8')
        return dialog_sequence

    @staticmethod
    def prune_rounds(context, num_rounds):
        start_segment = 1
        len_context = len(context)
        cur_rounds = (len(context) // 2) + 1
        l_index = 0
        if cur_rounds > num_rounds:
            # caption is not part of the final input
            l_index = len_context - (2 * num_rounds)
            start_segment = 0
        return context[l_index:], start_segment

    def val_dataset_bert_info(self, item_feature):
        samples, gt_relevance, gt_option_inds = self.generate_val_samples(item_feature)
        language = self.generate_val_language_info(samples)

        item_feature.update(language)
        item_feature['gt_relevance'] = gt_relevance
        item_feature['gt_option_inds'] = torch.LongTensor(gt_option_inds)
        item_feature['round_id'] = torch.LongTensor([item_feature['round_id']])
        item_feature['dialog'] = [item_feature['dialog']]

        return item_feature

    def train_dataset_bert_info(self, item_feature):
        utterances, utterances_random = self.generate_train_samples(item_feature)
        utterances, utterances_random = utterances[1:], utterances_random[1:]  # remove the caption in the beginning
        assert len(utterances) == len(utterances_random) == 10

        language = self.generate_train_language_info(utterances, utterances_random)
        item_feature.update(language)
        return item_feature

    def test_dataset_bert_info(self, item_feature):
        test_samples = self.generate_test_samples(item_feature)
        language = self.generate_test_language_info(test_samples)
        item_feature.update(language)

        return item_feature


class VisualDialogDenseInfoCpler(VisDiaInfoCpler):

    def __init__(self, cfg):
        super().__init__(cfg)

    def complete_info(self, item_feature, split='train'):
        cur_rnd_utterance = [self.tokenizer.encode(item_feature.caption)]
        cur_rounds = item_feature.round_id

        for rnd, utterance in enumerate(item_feature.dialog[:cur_rounds]):
            cur_rnd_utterance.append(self.tokenizer.encode(utterance['question']))
            if rnd != cur_rounds - 1:
                cur_rnd_utterance.append(self.tokenizer.encode(utterance['answer']))

        options_all = []
        for answer in item_feature.dialog[cur_rounds - 1]['answer_options']:
            cur_option = cur_rnd_utterance.copy()
            cur_option.append(self.tokenizer.encode(answer))
            options_all.append(cur_option)
            assert len(cur_option) == 2 * cur_rounds + 1

        gt_option = item_feature.dialog[cur_rounds - 1]['gt_index']

        tokens_all = []
        mask_all = []
        segments_all = []
        sep_indices_all = []
        hist_len_all = []

        for _, option in enumerate(options_all):
            option, start_segment = super().prune_rounds(option, self.visual_dialog_tot_rounds)
            tokens, segments, sep_indices, mask = encode_input(
                option,
                start_segment,
                self.CLS,
                self.SEP,
                self.MASK,
                max_seq_len=self.max_sequence_len,
                mask_prob=self.mask_probability)

            tokens_all.append(tokens)
            mask_all.append(mask)
            segments_all.append(segments)
            sep_indices_all.append(sep_indices)
            hist_len_all.append(torch.LongTensor([len(option) - 1]))

        tokens_all = torch.cat(tokens_all, 0)
        mask_all = torch.cat(mask_all, 0)
        segments_all = torch.cat(segments_all, 0)
        sep_indices_all = torch.cat(sep_indices_all, 0)
        hist_len_all = torch.cat(hist_len_all, 0)

        item = ItemFeature()
        item['tokens'] = tokens_all.unsqueeze(0)
        item['segments'] = segments_all.unsqueeze(0)
        item['sep_indices'] = sep_indices_all.unsqueeze(0)
        item['mask'] = mask_all.unsqueeze(0)
        item['hist_len'] = hist_len_all.unsqueeze(0)
        item['image_id'] = torch.LongTensor([item_feature.image_id])

        item['image_feat'] = item_feature.image_feat
        item['image_loc'] = item_feature.image_loc
        item['image_mask'] = item_feature.image_mask
        item['image_target'] = item_feature.image_target
        item['image_label'] = item_feature.image_label

        # add dense annotation fields
        item['gt_relevance_round_id'] = torch.LongTensor([cur_rounds])
        item['gt_relevance'] = torch.Tensor(item_feature['relevance'])
        item['gt_option'] = torch.LongTensor([gt_option])

        # add next sentence labels for training with the nsp loss as well
        nsp_labels = torch.ones(*tokens_all.unsqueeze(0).shape[:-1])
        nsp_labels[:, gt_option] = 0
        item['next_sentence_labels'] = nsp_labels.long()

        return item
