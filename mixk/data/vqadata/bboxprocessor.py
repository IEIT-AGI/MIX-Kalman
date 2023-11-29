from ..builder import PROCESSOR
from .vocabprocessor import VocabProcessor
import torch
from collections import OrderedDict
import collections


class Sample(OrderedDict):
    """Sample represent some arbitrary data. All datasets in  must return an
    object of type ``Sample``.

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


def build_bbox_tensors(infos, max_length):
    num_bbox = min(max_length, len(infos))

    # After num_bbox, everything else should be zero
    coord_tensor = torch.zeros((max_length, 4), dtype=torch.float)
    width_tensor = torch.zeros(max_length, dtype=torch.float)
    height_tensor = torch.zeros(max_length, dtype=torch.float)
    bbox_types = ['xyxy'] * max_length

    infos = infos[:num_bbox]
    sample = Sample()

    for idx, info in enumerate(infos):
        bbox = info['bounding_box']
        x = bbox.get('top_left_x', bbox['topLeftX'])
        y = bbox.get('top_left_y', bbox['topLeftY'])
        width = bbox['width']
        height = bbox['height']

        coord_tensor[idx][0] = x
        coord_tensor[idx][1] = y
        coord_tensor[idx][2] = x + width
        coord_tensor[idx][3] = y + height

        width_tensor[idx] = width
        height_tensor[idx] = height
    sample.coordinates = coord_tensor
    sample.width = width_tensor
    sample.height = height_tensor
    sample.bbox_types = bbox_types

    return sample


@PROCESSOR.register_module()
class BBoxProcessor(VocabProcessor):
    """Generates bboxes in proper format. Takes in a dict which contains "info"
    key which is a list of dicts containing following for each of the the
    bounding box.

    Example bbox input::

        {
            "info": [
                {
                    "bounding_box": {
                        "top_left_x": 100,
                        "top_left_y": 100,
                        "width": 200,
                        "height": 300
                    }
                },
                ...
            ]
        }


    This will further return a Sample in a dict with key "bbox" with last
    dimension of 4 corresponding to "xyxy". So sample will look like following:

    Example Sample::

        Sample({
            "coordinates": torch.Size(n, 4),
            "width": List[number], # size n
            "height": List[number], # size n
            "bbox_types": List[str] # size n, either xyxy or xywh.
            # currently only supports xyxy.
        })
    """

    def __init__(self, max_length, *args, **kwargs):
        self.lambda_fn = build_bbox_tensors
        self.max_length = max_length
        # self._init_extras(config)

    def __call__(self, item):
        info = item['info']
        if self.preprocessor is not None:
            info = self.preprocessor(info)

        return {'bbox': self.lambda_fn(info, self.max_length)}
