import collections
from collections import OrderedDict


class ItemFeature(OrderedDict):
    """ItemFeature is the basic stream form of data. All datasets in MIX should
    be changed into an object of ItemFeature.

    Args:
        init_dict (dict) : Dictionary to init ItemFeature
    """

    def __init__(self, init_dict=None):
        if init_dict is None:
            init_dict = {}
        super().__init__(init_dict)
        for key, value in init_dict.items():
            self[key] = value

    def __setattr__(self, key, value):
        if isinstance(value, collections.abc.Mapping):
            value = ItemFeature(value)
        self[key] = value

    def __setitem__(self, key, value):
        if isinstance(value, collections.abc.Mapping):
            value = ItemFeature(value)
        super().__setitem__(key, value)

    def __getattr__(self, key):
        try:
            return self[key]
        # except KeyError:
        #     raise AttributeError
        except KeyError:
            return None

    def fields(self):
        return list(self.keys())
