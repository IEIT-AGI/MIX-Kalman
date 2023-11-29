from collections import OrderedDict
from typing import Any, Dict
import collections

import numpy as np
import torch


class VQA2Dataset(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()
        self._load_npy(path)

    def _load_npy(self, path):
        self.db = np.load(path, allow_pickle=True)
        self.start_idx = 0

        if type(self.db) == dict:
            self.metadata = self.db.get('metadata', {})
            self.data = self.db.get('data', [])
        else:
            # TODO: Deprecate support for this
            self.metadata = {'version': 1}
            self.data = self.db
            # Handle old imdb support
            if 'image_id' not in self.data[0]:
                self.start_idx = 1

        if len(self.data) == 0:
            self.data = self.db

    def init_processors(self):
        # self.text_processor = Vocab
        self.text_processor = None

    def get(self, item):
        feature_path = item.get(self.feature_key, None)

        if feature_path is None:
            feature_path = self._get_feature_path_based_on_image(item)

        return self.from_path(feature_path)

    def _get_feature_path_based_on_image(self, item):
        image_path = self._get_attrs(item)[0]
        feature_path = '.'.join(image_path.split('.')[:-1]) + '.npy'
        return feature_path

    def from_path(self, path):
        assert isinstance(path, str)

        if 'genome' in path and path.endswith('.npy'):
            path = str(int(path.split('_')[-1].split('.')[0])) + '.npy'

        features, infos = self._get_image_features_and_info(path)

        item = {}
        for idx, image_feature in enumerate(features):
            item['image_feature_%s' % idx] = image_feature
            if infos is not None:
                # infos[idx].pop("cls_prob", None)
                item['image_info_%s' % idx] = infos[idx]

        return item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.load_item(idx)

    def load_item(self, idx):
        sample_info = self.data[idx]
        current_sample = Sample()

        if 'question_tokens' in sample_info:
            text_processor_argument = {
                'tokens': sample_info['question_tokens'],
                'text': sample_info['question_str'],
            }
        else:
            text_processor_argument = {'text': sample_info['question']}

        processed_question = self.text_processor(text_processor_argument)

        current_sample.text = processed_question['text']
        if 'input_ids' in processed_question:
            current_sample.update(processed_question)

        current_sample.question_id = torch.tensor(sample_info['question_id'], dtype=torch.int)

        if isinstance(sample_info['image_id'], int):
            current_sample.image_id = torch.tensor(sample_info['image_id'], dtype=torch.int)
        else:
            current_sample.image_id = sample_info['image_id']

        current_sample.text_len = torch.tensor(len(sample_info['question_tokens']), dtype=torch.int)

        if self._use_features is True:
            features = self.get(sample_info)
            if hasattr(self, 'transformer_bbox_processor'):
                features['image_info_0'] = self.transformer_bbox_processor(features['image_info_0'])
            current_sample.update(features)

        # Add details for OCR like OCR bbox, vectors, tokens here
        current_sample = self.add_ocr_details(sample_info, current_sample)
        # Depending on whether we are using soft copy this can add
        # dynamic answer space
        current_sample = self.add_answer_info(sample_info, current_sample)
        return current_sample


class BatchCollator:

    def __init__(self, dataset_name, dataset_type):
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type

    def __call__(self, batch):
        # Create and return sample list with proper name
        # and type set if it is already not a sample list
        # (case of batched iterators)
        sample_list = batch
        if (
                # Check if batch is a list before checking batch[0]
                # or len as sometimes batch is already SampleList
                isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], SampleList)):
            sample_list = batch[0]
        elif not isinstance(batch, SampleList):
            sample_list = SampleList(batch)

        sample_list.dataset_name = self._dataset_name
        sample_list.dataset_type = self._dataset_type
        return sample_list


class Sample(OrderedDict):
    """Sample represent some arbitrary data. All datasets in mixk must return
    an object of type ``Sample``.

    Args:
        init_dict (Dict): Dictionary to init ``Sample`` class with.

    Usage::

        >>> sample = Sample({"text": torch.tensor(2)})
        >>> sample.text.zero_()
        # Custom attributes can be added to ``Sample`` after initialization
        >>> sample.context = torch.tensor(4)
    """

    def __init__(self, init_dict=None):
        if init_dict is None:
            init_dict = {}
        super().__init__(init_dict)

    def __setattr__(self, key, value):
        if isinstance(value, collections.abc.Mapping):
            value = Sample(value)
        self[key] = value

    def __setitem__(self, key, value):
        if isinstance(value, collections.abc.Mapping):
            value = Sample(value)
        super().__setitem__(key, value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def fields(self):
        """Get current attributes/fields registered under the sample.

        Returns:
            List[str]: Attributes registered under the Sample.
        """
        return list(self.keys())


class SampleList(OrderedDict):
    """``SampleList`` is used to collate a list of ``Sample`` into a batch
    during batch preparation. It can be thought of as a merger of list of Dicts
    into a single Dict.

    If ``Sample`` contains an attribute 'text' of size (2) and there are 10 samples in
    list, the returned ``SampleList`` will have an attribute 'text' which is a tensor
    of size (10, 2).

    Args:
        samples (type): List of ``Sample`` from which the ``SampleList``
                        will be created.

    Usage::

        >>> sample_list = [
                Sample({"text": torch.tensor(2)}),
                Sample({"text": torch.tensor(2)})
            ]
        >>> sample_list.text
        torch.tensor([2, 2])
    """

    _TENSOR_FIELD_ = '_tensor_field'

    def __init__(self, samples=None):
        super().__init__(self)
        if samples is None:
            samples = []

        if len(samples) == 0:
            return

        if self._check_and_load_dict(samples):
            return
        # If passed sample list was in form of key, value pairs of tuples
        # return after loading these
        if self._check_and_load_tuple(samples):
            return

        fields = samples[0].keys()

        for field in fields:
            if isinstance(samples[0][field], torch.Tensor):
                size = (len(samples), *samples[0][field].size())
                self[field] = samples[0][field].new_empty(size)
                if self._get_tensor_field() is None:
                    self._set_tensor_field(field)
            else:
                self[field] = [None for _ in range(len(samples))]

            for idx, sample in enumerate(samples):
                # it should be a tensor but not a 0-d tensor
                if (isinstance(sample[field], torch.Tensor) and len(sample[field].size()) != 0
                        and sample[field].size(0) != samples[0][field].size(0)):
                    raise AssertionError('Fields for all samples must be equally sized. '
                                         '{} is of different sizes'.format(field))

                self[field][idx] = self._get_data_copy(sample[field])

            if isinstance(samples[0][field], collections.abc.Mapping):
                self[field] = SampleList(self[field])

    def _check_and_load_tuple(self, samples):
        if isinstance(samples[0], (tuple, list)) and isinstance(samples[0][0], str):
            for kv_pair in samples:
                self.add_field(kv_pair[0], kv_pair[1])
            return True
        else:
            return False

    def _check_and_load_dict(self, samples):
        if isinstance(samples, collections.abc.Mapping):
            for key, value in samples.items():
                self.add_field(key, value)
            return True
        else:
            return False

    def _fix_sample_type(self, samples):
        if not isinstance(samples[0], Sample):
            proper_samples = []
            for sample in samples:
                proper_samples.append(Sample(sample))
            samples = proper_samples
        return samples

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError('Key {} not found in the SampleList. '
                                 'Valid choices are {}'.format(key, self.fields()))
        fields = self.keys()

        if key in fields:
            return self[key]

        sample = Sample()

        for field in fields:
            sample[field] = self[field][key]

        return sample

    def get_device(self):
        field_tensor = self._get_tensor_field()
        return self[field_tensor].device

    def get_item_list(self, key):
        """Get ``SampleList`` of only one particular attribute that is present
        in the ``SampleList``.

        Args:
            key (str): Attribute whose ``SampleList`` will be made.

        Returns:
            SampleList: SampleList containing only the attribute value of the key
            which was passed.
        """
        sample = self[key]

        return SampleList([sample])

    def copy(self):
        """Get a copy of the current SampleList.

        Returns:
            SampleList: Copy of current SampleList.
        """
        sample_list = SampleList()

        fields = self.fields()

        for field in fields:
            sample_list.add_field(field, self[field])

        return sample_list

    def fields(self):
        """Get current attributes/fields registered under the SampleList.

        Returns:
            List[str]: list of attributes of the SampleList.
        """
        return list(self.keys())

    def get_fields(self, fields):
        """Get a new ``SampleList`` generated from the current ``SampleList``
        but contains only the attributes passed in `fields` argument.

        Args:
            fields (List[str]): Attributes whose ``SampleList`` will be made.

        Returns:
            SampleList: SampleList containing only the attribute values of the fields
            which were passed.
        """
        current_fields = self.fields()

        return_list = SampleList()

        for field in fields:
            if field not in current_fields:
                raise AttributeError('{} not present in SampleList. '
                                     'Valid choices are {}'.format(field, current_fields))
            return_list.add_field(field, self[field])

        return return_list

    def get_field(self, field):
        """Get value of a particular attribute.

        Args:
            field (str): Attribute whose value is to be returned.
        """
        return self[field]

    def _get_data_copy(self, data):
        # if isinstance(data, torch.Tensor):
        #     copy_ = data.clone()
        # else:
        #     copy_ = deepcopy(data)
        # return copy_
        return data

    def _get_tensor_field(self):
        return self.__dict__.get(SampleList._TENSOR_FIELD_, None)

    def _set_tensor_field(self, value):
        self.__dict__[SampleList._TENSOR_FIELD_] = value

    def get_batch_size(self):
        """Get batch size of the current ``SampleList``.

        There must be a tensor
        be a tensor present inside sample list to use this function.
        Returns:
            int: Size of the batch in ``SampleList``.
        """
        tensor_field = self._get_tensor_field()
        assert tensor_field is not None, 'There is no tensor yet in SampleList'

        return self[tensor_field].size(0)

    def add_field(self, field, data):
        """Add an attribute ``field`` with value ``data`` to the SampleList.

        Args:
            field (str): Key under which the data will be added.
            data (object): Data to be added, can be a ``torch.Tensor``, ``list``
                         or ``Sample``
        """
        fields = self.fields()
        tensor_field = self._get_tensor_field()

        if (len(fields) != 0 and isinstance(data, torch.Tensor) and len(data.size()) != 0 and tensor_field is not None
                and data.size(0) != self[tensor_field].size(0)):
            raise AssertionError('A tensor field to be added must '
                                 'have same size as existing tensor '
                                 'fields in SampleList. '
                                 'Passed size: {}, Required size: {}'.format(len(data), len(self[tensor_field])))

        if isinstance(data, collections.abc.Mapping):
            self[field] = SampleList(data)
        else:
            self[field] = self._get_data_copy(data)

            if isinstance(self[field], torch.Tensor) and tensor_field is None:
                self._set_tensor_field(field)

    def to(self, device, non_blocking=True):
        """Similar to ``.to`` function on a `torch.Tensor`. Moves all of the
        tensors present inside the ``SampleList`` to a particular device. If an
        attribute's value is not a tensor, it is ignored and kept as it is.

        Args:
            device (str|torch.device): Device on which the ``SampleList`` should
                                       moved.
            non_blocking (bool): Whether the move should be non_blocking. Default: True

        Returns:
            SampleList: a SampleList moved to the ``device``.
        """
        fields = self.keys()
        sample_list = self.copy()
        if not isinstance(device, torch.device):
            if not isinstance(device, str):
                raise TypeError("device must be either 'str' or " "'torch.device' type, {} found".format(type(device)))
            device = torch.device(device)

        for field in fields:
            if hasattr(sample_list[field], 'to'):
                sample_list[field] = sample_list[field].to(device, non_blocking=non_blocking)

        return sample_list

    def pin_memory(self):
        """In custom batch object, we need to define pin_memory function so
        that PyTorch can actually apply pinning.

        This function just individually pins all of the tensor fields
        """
        fields = self.keys()

        for field in fields:
            if hasattr(self[field], 'pin_memory'):
                # This will also handle nested sample list recursively
                self[field] = self[field].pin_memory()

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Converts a sample list to dict, this is useful for TorchScript and
        for other internal API unification efforts.

        Returns:
            Dict[str, Any]: A dict representation of current sample list
        """
        sample_dict = {}
        fields = self.keys()

        for field in fields:
            # Handle nested sample list recursively
            if hasattr(self[field], 'to_dict'):
                sample_dict[field] = self[field].to_dict()
            else:
                sample_dict[field] = self[field]

        return sample_dict
