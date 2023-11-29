from typing import Dict, List

import numpy as np
import torch
from allennlp.data.fields import SequenceLabelField
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.tokenizers import Token
from allennlp.nn import util
from matplotlib import path
from overrides import overrides
import numpy

GENDER_NEUTRAL_NAMES = [
    'Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall', 'Peyton', 'Skyler',
    'Frankie', 'Pat', 'Quinn'
]

# TokenList = List[TokenType]  # pylint: disable=invalid-name


# This will work for anything really
class BertField(SequenceField[Dict[str, torch.Tensor]]):
    """A class representing an array, which could have arbitrary dimensions.

    A batch of these arrays are padded to the max dimension length in the batch for each dimension.
    """

    def __init__(self, tokens: List[Token], embs: np.ndarray, padding_value: int = 0, token_indexers=None) -> None:
        self.tokens = tokens
        self.embs = embs
        self.padding_value = padding_value

        if len(self.tokens) != self.embs.shape[0]:
            raise ValueError('The tokens you passed into the BERTField, {} '
                             "aren't the same size as the embeddings of shape {}".format(self.tokens, self.embs.shape))
        assert len(self.tokens) == self.embs.shape[0]

    @overrides
    def sequence_length(self) -> int:
        return len(self.tokens)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_tokens': self.sequence_length()}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        num_tokens = padding_lengths['num_tokens']

        new_arr = numpy.ones((num_tokens, self.embs.shape[1]), dtype=numpy.float32) * self.padding_value
        new_arr[:self.sequence_length()] = self.embs

        tensor = torch.from_numpy(new_arr)
        return {'bert': tensor}

    @overrides
    def empty_field(self):
        return BertField([], numpy.array([], dtype='float32'), padding_value=self.padding_value)

    @overrides
    def batch_tensors(self, tensor_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # pylint: disable=no-self-use
        # This is creating a dict of {token_indexer_key: batch_tensor} for each token indexer used
        # to index this field.
        return util.batch_tensor_dicts(tensor_list)

    def __str__(self) -> str:
        return f'BertField: {self.tokens} and  {self.embs.shape}.'


def _fix_tokenization(tokenized_sent, bert_embs, old_det_to_new_ind, obj_to_type, token_indexers, pad_ind=-1):
    """Turn a detection list into what we want: some text, as well as some
    tags.

    :param tokenized_sent: Tokenized sentence with detections collapsed to a list.
    :param old_det_to_new_ind: Mapping of the old ID -> new ID (which will be used as the tag)
    :param obj_to_type: [person, person, pottedplant] indexed by the old labels
    :return: tokenized sentence
    """

    new_tokenization_with_tags = []
    for tok in tokenized_sent:
        if isinstance(tok, list):
            for int_name in tok:
                obj_type = obj_to_type[int_name]
                new_ind = old_det_to_new_ind[int_name]
                if new_ind < 0:
                    raise ValueError("Oh no, the new index is negative! that means it's invalid. {} {}".format(
                        tokenized_sent, old_det_to_new_ind))
                text_to_use = GENDER_NEUTRAL_NAMES[new_ind %
                                                   len(GENDER_NEUTRAL_NAMES)] if obj_type == 'person' else obj_type
                new_tokenization_with_tags.append((text_to_use, new_ind))
        else:
            new_tokenization_with_tags.append((tok, pad_ind))

    text_field = BertField([Token(x[0]) for x in new_tokenization_with_tags], bert_embs, padding_value=0)
    tags = SequenceLabelField([x[1] for x in new_tokenization_with_tags], text_field)
    return text_field, tags


def _spaced_points(low, high, n):
    """We want n points between low and high, but we don't want them to touch
    either side."""
    padding = (high - low) / (n * 2)
    return np.linspace(low + padding, high - padding, num=n)


def make_mask(mask_size, box, polygons_list):
    """
    Mask size: int about how big mask will be
    box: [x1, y1, x2, y2, conf.]
    polygons_list: List of polygons that go inside the box
    """
    mask = np.zeros((mask_size, mask_size), dtype=np.bool)

    xy = np.meshgrid(_spaced_points(box[0], box[2], n=mask_size), _spaced_points(box[1], box[3], n=mask_size))
    xy_flat = np.stack(xy, 2).reshape((-1, 2))

    for polygon in polygons_list:
        polygon_path = path.Path(polygon)
        mask |= polygon_path.contains_points(xy_flat).reshape((mask_size, mask_size))
    return mask.astype(np.float32)
