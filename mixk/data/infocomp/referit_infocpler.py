from .refcoco2_infocpler import RefCOCOInfoCpler


class ReferitInfoCpler(RefCOCOInfoCpler):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.vocab_name = cfg.get('vocab_name', 'vocabulart_100k')
        self.vocab_path = self._get_atr_of_atr(cfg, 'mix_vocab', self.vocab_name)
        self.vocab_answer_name = cfg.get('vocab_answer_name', 'answers_vqa')
        self.vocab_answer_path = self._get_atr_of_atr(cfg, 'mix_vocab', self.vocab_answer_name)

    # inheriting func
    # def complete_info(self, item_feature: ItemFeature):
    #    item_feature.img = self.transform(item_feature.img)
    #    phrases = item_feature.phrase
    #    tokenss = [self.tokenizer.tokenize(phrase.strip()) for phrase in phrases]
    #    tokens_r = [self._CLS_TOKEN]
    #    input_type_ids = [0]
    #    for i, tokens in enumerate(tokenss):
    #        tokens_r += tokens
    #        tokens_r += [self._SEP_TOEKN]
    #        input_type_ids += [i] * (len(tokens) + 1)
    #    input_ids = self.tokenizer.convert_tokens_to_ids(tokens_r)

    #    input_mask = [1] * len(input_ids)
    #    while len(input_ids) < self.default_max_length:
    #        input_ids.append(0)
    #        input_mask.append(0)
    #        input_type_ids.append(0)

    #    input_ids = np.array(input_ids[:self.default_max_length])
    #    input_mask = np.array(input_mask[:self.default_max_length])
    #    input_type_ids = np.array(input_type_ids[:self.default_max_length])

    #    item_feature.input_ids = input_ids
    #    item_feature.input_mask = input_mask
    #    item_feature.input_type_ids = input_type_ids

    #    return item_feature
