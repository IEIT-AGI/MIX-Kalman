import numpy as np

from ..utils.stream import ItemFeature
from ..utils.tokenization import BertTokenizer
from .base_reader import mixkDataReader


class GQAReader(mixkDataReader):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def __len__(self):
        return len(self.mix_annotations)

    def __getitem__(self, idx):
        annotation = self.mix_annotations[idx]
        features = self.feature_obj[idx]

        item_feature = ItemFeature(annotation)

        item_feature.error = False
        item_feature.tokens = self.tokenize_gqa(annotation['question_str'])
        item_feature.tokens_len = len(item_feature.tokens)
        item_feature.img_id = annotation['image_id']

        if features is None:
            item_feature.error = True
            item_feature.feature = np.random.random((100, 2048))
        else:
            item_feature.update(features)

        return item_feature

    @staticmethod
    def tokenize_gqa(sentence, ignored_punct=['?', '!', '\\', '/', ')', '('], kept_punct=['.', ',', ';', ':']):
        sentence = sentence.lower()
        for punct in kept_punct:
            sentence = sentence.replace(punct, ' ' + punct + ' ')
        for punct in ignored_punct:
            sentence = sentence.replace(punct, '')
        tokens = sentence.split()
        return tokens
