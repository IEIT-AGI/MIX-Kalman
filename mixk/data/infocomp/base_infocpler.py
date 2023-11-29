from collections import defaultdict

import numpy as np
import torch
from fasttext import load_model

from ..utils.phoc import phoc
from ..utils.tokenization import BertTokenizer


class BaseInfoCpler(object):

    def __init__(self, cfg):
        # logger = logging.getLogger(__name__)

        self.cfg = cfg
        self.if_bert = cfg.if_bert
        self._init_tokens()

        self.max_seq_length = cfg.get('max_seg_lenth', 14)
        self.max_ocr_length = cfg.get('max_seq_length', 50)
        self.word_mask_ratio = cfg.get('word_mask_ratio', 0.15)

        self.vocab_name = cfg.get('vocab_name', 'vocabulart_100k')
        self.vocab_path = self._get_atr_of_atr(cfg, 'mix_vocab', self.vocab_name)

        self.vocab_answer_name = cfg.get('vocab_answer_name', 'answers_vqa')
        self.vocab_answer_path = self._get_atr_of_atr(cfg, 'mix_vocab', self.vocab_answer_name)

        self.glove_name = cfg.get('glove_name', 'glove6b300d')
        self.glove_weights_path = self._get_atr_of_atr(cfg, 'glove_weights', self.glove_name)

        self.fasttext_name = cfg.get('fasttext_name', 'wiki300d1m')
        self.fasttext_weights_path = self._get_atr_of_atr(cfg, 'fasttext_weights', self.fasttext_name)
        self.fasttext_model = None

        self.load_glove_weights()
        # self.load_fasttext_weights()
        self.load_vocab()
        self.init_phoc()

    def compute_answers_scores(self, answers_indices):
        """Generate VQA based answer scores for answers_indices.

        Args:
            answers_indices (torch.LongTensor): tensor containing indices of the answers

        Returns:
            torch.FloatTensor: tensor containing scores.
        """
        scores = torch.zeros(len(self.qa_ans2id), dtype=torch.float)
        gt_answers = list(enumerate(answers_indices))
        unique_answers = set(answers_indices.tolist())

        for answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            avg_acc = sum(accs) / len(accs)
            if answer != 0:
                scores[int(answer)] = avg_acc
        return scores

    def _increase_to_ten(self, tokens):
        while len(tokens) < self.DEFAULT_NUM_ANSWERS:
            tokens += tokens[:self.DEFAULT_NUM_ANSWERS - len(tokens)]
        return tokens

    def load_glove_weights(self):
        if self.glove_weights_path is None:
            return
        glove = torch.load(self.glove_weights_path)
        self.glove_vocabs = glove[0]
        self.glove_vocab_dict = glove[1]
        self.glove_weights = glove[2]
        self.glove_weights_dim = len(self.glove_weights[0])

    def get_glove_single_word(self, word):
        try:
            return self.glove_weights[self.glove_vocab_dict[word]]
        except Exception:
            return ([0] * 300).copy()

    def get_glove_single_id(self, id):
        if id == self.pad_idx:
            return torch.zeros((300, ))
        try:
            return self.glove_weights[id]
        except Exception:
            return torch.zeros((300, ))

    def get_tokens_glove_vectors(self, tokens):
        vector = torch.full((self.max_ocr_length, self.glove_weights_dim), fill_value=self.PAD_INDEX, dtype=torch.float)
        for idx, token in enumerate(tokens):
            vector[idx] = torch.tensor(self.get_glove_single_word(token.lower()))
        return vector

    def get_tokens_order_vectors(self, tokens):
        vector = torch.full((self.max_ocr_length, self.glove_weights_dim), fill_value=self.PAD_INDEX, dtype=torch.float)
        vector[:len(tokens), :len(tokens)] = torch.from_numpy(np.eye(len(tokens), dtype=np.float32))
        return vector

    def get_tokens_phoc_vectors(self, tokens):
        vector = torch.full((self.max_ocr_length, self.phoc_weights_dim), fill_value=self.PAD_INDEX, dtype=torch.float)
        for idx, token in enumerate(tokens):
            vector[idx] = torch.tensor(phoc(token))[0]
        return vector

    def load_fasttext_weights(self):
        if self.fasttext_weights_path is None or self.fasttext_model is not None:
            return
        self.fasttext_model = load_model(self.fasttext_weights_path)

    def get_fasttext_single_word(self, word):
        return np.mean([self.fasttext_model.get_word_vector(w) for w in word.split(' ')], axis=0)

    def get_tokens_fasttext_vectors(self, tokens):
        vector = torch.full((self.max_ocr_length, self.fasttext_model.get_dimension()),
                            fill_value=self.PAD_INDEX,
                            dtype=torch.float)
        tokens = [self.word_tokenize(tmp) for tmp in tokens]
        for idx, token in enumerate(tokens):
            vector[idx] = torch.from_numpy(self.get_fasttext_single_word(token))
        return vector

    def word_tokenize(self, word, remove=None):
        if remove is None:
            remove = [',', '?']
        word = word.lower()

        for item in remove:
            word = word.replace(item, '')
        word = word.replace("'s", " 's")

        return word.strip()

    def load_vocab(self):
        if self.vocab_answer_path is None:
            return
        with open(self.vocab_answer_path) as f:
            raw_qa_vocab = f.readlines()
        self.qa_id2ans = [t.strip() for t in raw_qa_vocab]
        self.qa_ans2id = {k: v for v, k in enumerate(self.qa_id2ans)}
        self.DEFAULT_NUM_ANSWERS = 10

        self.word_dict = {}
        self.itos = {}

        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN
        self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
        self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
        self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX

        index = len(self.itos.keys())
        self.total_predefined = len(self.itos.keys())
        with open(self.vocab_path, 'r') as f:
            for line in f:
                self.itos[index] = line.strip()
                self.word_dict[line.strip()] = index
                index += 1

        self.stoi = defaultdict(self.get_unk_index)
        self.stoi.update(self.word_dict)

    def get_unk_index(self):
        return self.UNK_INDEX

    def _init_tokens(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.PAD_TOKEN = '<pad>'
        self.SOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'
        self.UNK_TOKEN = '<unk>'
        self.PAD_INDEX = 0
        self.SOS_INDEX = 1
        self.EOS_INDEX = 2
        self.UNK_INDEX = 3
        self._MASK_TOKEN = '[MASK]'
        self._SEP_TOEKN = '[SEP]'
        self._CLS_TOKEN = '[CLS]'
        self._PAD_TOKEN = '[PAD]'
        self.pad_idx = self.tokenizer.vocab[self._PAD_TOKEN]

    def init_phoc(self):
        self.phoc_weights_dim = len(phoc('0')[0])

    def _get_normalized_from_bbox(self, bbox, img_h, img_w):
        assert img_h > 0 and img_w > 0, ('Height and Width need to be positive.')
        return bbox / (np.array([img_w, img_h, img_w, img_h]))

    def _get_bbox_from_normalized(self, normalized, img_h, img_w):
        assert img_h > 0 and img_w > 0, ('Height and Width need to be positive.')
        return normalized * (np.array([img_w, img_h, img_w, img_h]))

    def _get_atr_of_atr(self, cfg, at1, at2):
        try:
            out = cfg.get(at1).get(at2)
            return out
        except Exception:
            return None

    def info_extend(self, length, *to_be_extend):
        for info, value in to_be_extend:
            info.extend([value] * length)
