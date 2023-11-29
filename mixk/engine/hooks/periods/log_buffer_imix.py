from collections import defaultdict, namedtuple
from contextlib import contextmanager

import torch

from mixk.utils.config import mixkEasyDict
from mixk.utils.history_buffer import HistoryBuffer

Image = namedtuple('Image', ('name', 'data', 'iter_idx'))


class VisualizeImage:

    def __init__(self):
        self._images = []

    def append(self, name: str, data: torch.Tensor, iter_idx: int = 0):
        self._images.append(Image(name, data, iter_idx))

    def clear(self):
        self._images = []

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, new_imgs):
        self._images = new_imgs

    def __len__(self):
        return len(self._images)


class Histogram:

    def __init__(self, bins: int = 800):
        self._histograms = []
        self._bins = bins

    def append(self, name: str, data: torch.Tensor, iter_idx: int = 0):
        self._histograms.append(self._convert(name, data, iter_idx))

    def _convert(self, name, data, iter_idx):
        ht_min, ht_max, ht_counts = data.min().item(), data.max().item(), torch.histc(data, self._bins)
        lins = torch.linspace(ht_min, ht_max, steps=self._bins + 1, dtype=torch.float32)
        histogram_raw_data = dict(
            tag=name,
            min=ht_min,
            max=ht_max,
            sum=float(data.sum()),
            sum_squares=float(torch.sum(data**2)),
            num=len(data),
            bucket_limits=lins[1:].tolist(),
            bucket_counts=ht_counts.tolist(),
            global_step=iter_idx,
        )
        return histogram_raw_data

    def clear(self):
        self._histograms = []

    @property
    def histograms(self):
        return self._histograms

    @histograms.setter
    def histograms(self, new_hists: list):
        self._histograms = new_hists

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, new_bins: int):
        self._bins = new_bins

    def __len__(self):
        return len(self._histograms)


class TrainInfo:

    def __init__(self,
                 iter: int = 0,
                 epoch: int = 0,
                 epoch_inner_iter: int = 0,
                 single_epoch_iters: int = 0,
                 is_by_epoch: bool = True):
        self.iter = iter
        self.epoch = epoch
        self.epoch_inner_iter = epoch_inner_iter
        self.single_epoch_iters = single_epoch_iters
        self.is_by_epoch = is_by_epoch

    @staticmethod
    def has_name(name):
        return name in ['iter', 'epoch', 'epoch_inner_iter', 'single_epoch_iters', 'is_by_epoch']

    def __setattr__(self, name, value):
        if self.has_name(name):
            return super().__setattr__(name, value)
        else:
            raise AttributeError('there is no such {} Attribute in TrainInfo'.format(name))


class VisualizeScalar:

    def __init__(self):
        self._scalars = mixkEasyDict()
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = mixkEasyDict()

    def clear_scalars(self):
        self._scalars = mixkEasyDict()

    @staticmethod
    def has_name(name):
        attr_list = ['_scalars', '_history', '_smoothing_hints']
        return name in attr_list or VisualizeScalar._underline_prefix(name) in attr_list

    @staticmethod
    def _underline_prefix(name):
        return '_' + name

    @property
    def scalars(self):
        return self._scalars

    @property
    def history(self):
        return self._history

    @property
    def smoothing_hits(self):
        return self._smoothing_hints

    def add_single(self, name, data, is_smooth_hint: bool = True, idx: int = 0):
        h = self._history[name]
        h.update(float(data), idx)

        self._scalars[name] = data
        is_exist = self._smoothing_hints.get(name)
        if is_exist is not None:
            assert (is_smooth_hint == is_exist), ValueError(
                'Scalar {} should be add with  a different smoothing_hits'.format(name))
        else:
            self._smoothing_hints[name] = is_smooth_hint

    def add_some(self, *, is_smooth_hint: bool = True, iter_idx: int = 0, **kwargs):
        for k, v in kwargs.items():
            self.add_single(k, v, is_smooth_hint=is_smooth_hint, idx=iter_idx)

    def get_history_with_name(self, name):
        h = self._history.get(name, None)
        assert h, 'no key {} in historys! '.format(name)
        return h

    def get_scalars_with_smooth(self, window_size=30):
        result = mixkEasyDict()
        for key, value in self._scalars.items():
            result[key] = self._history[key].median(window_size) if self._smoothing_hints[key] else value

        return result


_CURRENT_LOG_BUFFER_STACK = []


def get_log_buffer():
    assert len(
        _CURRENT_LOG_BUFFER_STACK), "get_log_buffer() has to be called inside a 'with LogBufferStorage(...)' context!"
    return _CURRENT_LOG_BUFFER_STACK[-1]


class LogBufferStorage:

    def __init__(self, start_iter=0, single_epoch_iters=0, start_epoch=0, by_epoch=True):
        self.train_info = TrainInfo(
            iter=start_iter, epoch=start_epoch, single_epoch_iters=single_epoch_iters, is_by_epoch=by_epoch)
        self.vis_data = VisualizeImage()
        self.histograms = Histogram()
        self.current_prefix = ''

        self.vis_scalars = VisualizeScalar()

    def put_scalar(self, name, data, smoothing_hint=True, is_epoch=False):
        idx = self.train_info.epoch if is_epoch else self.train_info.iter
        self.vis_scalars.add_single(name, data, smoothing_hint, idx=idx)

    def put_scalars(self, *, smoothing_hint: bool = True, **kwargs):
        self.vis_scalars.add_some(is_smooth_hint=smoothing_hint, iter_idx=self.train_info.iter, **kwargs)

    def put_image(self, name, data):
        """Data to be visualized on tensorboard.

        Args:
            name (str): data identifier
            data (torch.Tensor or numpy.array): image data , An `uint8` or `float`
                Tensor of shape `[channel, height, width]` where `channel` is
                3. The  format of the image should be RGB.,where the range of element is [0,1](float)
                or [0,255](uint8).
        """
        self.vis_data.append(name, data, self.train_info.iter)

    def clear_images(self):
        self.vis_data.clear()

    def put_histogram(self, name, data):
        self.histograms.append(name, data, self.train_info.iter)

    def clear_histograms(self):
        self.histograms.clear()

    def has_name(self, name):
        log_buffer_attr = ['train_info', 'vis_data', 'histograms', 'current_prefix', 'vis_scalars']
        return name in log_buffer_attr

    def __getattr__(self, name):
        if self.has_name(name):
            return super().__getattr__(name)
        elif self.train_info.has_name(name):
            return self.train_info.__getattribute__(name)
        elif self.vis_scalars.has_name(name):
            return self.vis_scalars.__getattribute__(name)

    def __setattr__(self, key, value):
        if self.has_name(key):
            super().__setattr__(key, value)
        elif self.train_info.has_name(key):
            self.train_info.__setattr__(key, value)
        elif self.vis_scalars.has_name(key):
            self.vis_scalars.__setattr__(key, value)

    def step(self):
        self.train_info.iter += 1
        self.vis_scalars.clear_scalars()

    def epoch_step(self):
        self.train_info.epoch += 1

    def epoch_iter(self, inner_iter):
        self.train_info.epoch_inner_iter = inner_iter + 1

    def __enter__(self):
        _CURRENT_LOG_BUFFER_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_LOG_BUFFER_STACK[-1] == self
        _CURRENT_LOG_BUFFER_STACK.pop()

    @property
    def by_epoch(self):
        return self.train_info.is_by_epoch

    def history(self, name=None):
        ret = self.vis_scalars.history.get(name, None)
        if ret is None:
            raise KeyError('No history metric available for {}!'.format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self.vis_scalars.history

    def latest_with_smoothing_hint(self, window_size=20):
        return self.vis_scalars.get_scalars_with_smooth(window_size)

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self.current_prefix
        self.current_prefix = name.rstrip('/') + '/'
        yield
        self.current_prefix = old_prefix


class LogBufferWriter:
    """Used to write LogBufferStorage content."""

    def write(self):
        self.get_buffer_data()
        self.process_buffer_data()

    def close(self):
        pass

    def get_buffer_data(self):
        self.log_buffer = get_log_buffer()

    def process_buffer_data(self):
        raise NotImplementedError
